"""
Live seller world -- registry built by periodically polling a Redis HASH
that real sellers' status gets written into, one field per seller.

Same public interface the simulated and stream-tailing versions before it
had (`sellers`, `capacity`, `carbon`, `latency_for_buyer`,
`initial_remaining`, `snapshot()`), so nothing in algorithms/ or
workers/ needs to change for this swap -- this file is still the only
thing that had to change.

*** TRANSPORT: HASH, POLLED -- NOT A STREAM, TAILED ***
Earlier versions of this file tailed a Redis Stream (`seller-updates`)
via XRANGE catch-up + XREAD. That stream still exists and still carries
one event per seller update, but this file now reads a different key
instead: SELLABLE_RESOURCES_HASH (default "sellable_resources"), a plain
Redis HASH where each field is one seller's node id and each value is
that seller's full JSON status document. A single HGETALL always returns
the complete, current state of every seller in one call -- there's no
history to replay, so no catch-up step is needed the way a stream would
require one, and read cost scales with the number of sellers, not with
how long the system has been running.

    docker exec redis redis-cli HGETALL sellable_resources
    clab-nebula-extended-serf36
    {"node": "clab-nebula-extended-serf36", "collected_at": "...", "sellable": {...}}
    clab-nebula-extended-serf50
    {"node": "clab-nebula-extended-serf50", "collected_at": "...", "sellable": {...}}
    ...

*** REAL SELLER MESSAGE SCHEMA (the JSON value of each hash field) ***

    {
      "node": "clab-nebula-extended-serf1",
      "collected_at": "2026-09-28T14:32:10Z",
      "sellable": {
        "cpu": 238.0,
        "ram": 1902.0,
        "GPU": 0.0,
        "storage": 687.0
      }
    }

    node            The seller/node id. Falls back to the hash FIELD KEY
                    itself if the document omits it (the two are
                    expected to match, per the real data).
    collected_at    optional ISO8601 timestamp of when the seller took
                    this reading. Stored as metadata only -- NOT used to
                    judge staleness (see below), to avoid trusting
                    clocks across nodes.
    sellable        cpu/ram/GPU/storage -- CURRENTLY AVAILABLE capacity
                    for that resource (not total/max -- see the
                    capacity-authority note below). Field names are
                    matched case-insensitively and aliased onto our
                    internal resource names (ram -> mem, GPU -> gpu,
                    storage -> storage, cpu -> cpu); see
                    _RESOURCE_ALIASES. A missing resource defaults to 0,
                    meaning "this seller doesn't currently have any of
                    this resource free."

*** STALENESS: CHANGE-DETECTION, NOT "STILL PRESENT" AND NOT collected_at ***
Because every poll re-reads the *entire* hash, a seller that's simply
sitting there with a stale, unchanging value would look "present" on
every single poll forever -- "still in the hash" is not a liveness
signal the way "we just received a message about it" was for the old
stream-tailing design. So SellerWorld tracks, per seller, the last time
its raw JSON value actually *changed* (a plain string comparison against
what we saw last poll), not the last time we merely saw it. A seller
whose value hasn't changed in SELLER_STALE_AFTER_SECONDS is dropped, same
as before. This deliberately avoids trusting `collected_at` (i.e. the
seller's own clock) for the same clock-skew reason the old design did --
we still only trust our own local sense of time, just measured against
"did the content change" instead of "did a message arrive".
Separately, if a seller's field disappears from the hash entirely on a
given poll, it's removed immediately -- that's an unambiguous offline
signal, if whatever maintains this hash prunes offline nodes; if it
doesn't, the change-detection expiry above is the fallback that still
catches it.

*** CAPACITY AUTHORITY: "seller's next update overwrites ours" ***
Every fresh value for a seller REPLACES this registry's number for that
seller outright (last-write-wins, no averaging/merging). Each algorithm
processor keeps its own private `remaining` dict that it still decrements
locally between refreshes (via algorithms.common.resync_remaining(),
called at the top of every decision), so a single run still shows a
consistent, depleting picture within one algorithm's own sequence of
decisions -- but the moment a seller's hash entry changes, that seller's
capacity snaps back to whatever was just reported, discarding whatever
we'd locally guessed. Nothing here ever writes back to the seller side,
so this is a read-only consumer.

Note this doesn't fully close the race where a seller's real capacity
changes in the gap between "we last polled" and "an algorithm commits a
decision" -- no read-then-decide design can, without an actual
reserve/confirm handshake with the seller, which is a materially bigger
piece of scope than this file. SELLER_POLL_INTERVAL_SECONDS controls how
wide that gap can get; resync_remaining() being called immediately
before every decision keeps each algorithm's exposure as tight as the
poll interval allows.

If your real sellers instead publish TOTAL/max capacity (not currently-
available), you have two options: (a) have whatever writes this hash
compute and publish "available = total - in_use" itself, which is the
cleanest fix and keeps this file unchanged, or (b) change what this file
treats as ground truth by tracking total capacity separately and letting
only OUR OWN commits (never the seller's report) reduce `remaining` --
ask if you want that version instead, since it's a materially different
reconciliation policy than the one implemented here.
"""
import json
import threading
import time

from seller_workload_opt.v3.common.config import (
    SELLABLE_RESOURCES_HASH, SELLER_POLL_INTERVAL_SECONDS,
    SELLER_STALE_AFTER_SECONDS, DEFAULT_CARBON, DEFAULT_LATENCY,
    RESOURCES,
)
from seller_workload_opt.v3.common.logging_setup import get_logger
from seller_workload_opt.v3.common.redis_client import get_redis

log = get_logger("world")

# Real-world field names -> our internal resource names. Matched
# case-insensitively (keys here are already lowercase; the lookup
# lowercases whatever it's given).
_RESOURCE_ALIASES = {
    "cpu": "cpu",
    "mem": "mem",
    "ram": "mem",
    "memory": "mem",
    "gpu": "gpu",
    "storage": "storage",
    "disk": "storage",
}


def _resource_key(raw_key: str):
    return _RESOURCE_ALIASES.get(str(raw_key).strip().lower())


def _extract_capacity(source: dict) -> dict:
    """
    Scans a flat dict (the "sellable" sub-object) for keys that map onto
    one of our resource names, case-insensitively. Returns a full
    {r: 0.0, ...} dict for every entry in RESOURCES so callers never have
    to guard for a missing key.
    """
    capacity = {r: 0.0 for r in RESOURCES}
    for k, v in source.items():
        mapped = _resource_key(k)
        if mapped is None or mapped not in capacity:
            continue
        try:
            capacity[mapped] = float(v)
        except (TypeError, ValueError):
            pass
    return capacity


def _parse_seller_entry(field_key: str, raw_value: str):
    """
    field_key is the HASH field name (the node id, per HGETALL). raw_value
    is that field's value -- the seller's JSON document, e.g.
    {"node": "...", "collected_at": "...", "sellable": {...}}. Returns
    {seller_id, capacity, carbon, location, region, collected_at} or None
    if unusable (not valid JSON, not an object, or no node id at all) --
    caller logs and skips it.
    """
    try:
        doc = json.loads(raw_value)
    except (TypeError, ValueError):
        return None
    if not isinstance(doc, dict):
        return None

    seller_id = doc.get("node") or field_key
    if not seller_id:
        return None

    sellable = doc.get("sellable")
    capacity = _extract_capacity(sellable) if isinstance(sellable, dict) else {r: 0.0 for r in RESOURCES}

    # carbon/region/lat/lon aren't part of the real seller schema (sellers
    # don't report these) -- accepted at the top level of the document in
    # case a bridge ever adds them, defaulting otherwise.
    raw_carbon = doc.get("carbon")
    try:
        carbon = float(raw_carbon) if raw_carbon not in (None, "") else DEFAULT_CARBON
    except (TypeError, ValueError):
        carbon = DEFAULT_CARBON

    raw_lat, raw_lon = doc.get("lat"), doc.get("lon")
    location = None
    if raw_lat not in (None, "") and raw_lon not in (None, ""):
        try:
            location = (float(raw_lat), float(raw_lon))
        except (TypeError, ValueError):
            location = None
    region = doc.get("region") or None
    collected_at = doc.get("collected_at") or None

    return {
        "seller_id": seller_id, "capacity": capacity, "carbon": carbon,
        "location": location, "region": region, "collected_at": collected_at,
    }


class SellerWorld:
    """
    Live registry of sellers built by periodically polling
    SELLABLE_RESOURCES_HASH. One instance per process (API and each of
    the 4 workers each run their own, via get_world() below) -- every
    process independently reads the full hash, since this is reference
    data everyone needs completely, not a work queue to divide up.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._sellers = {}   # seller_id -> {capacity, carbon, location, region, collected_at, last_seen, _raw}
        self._r = get_redis()
        self._stop = False

        self._refresh_once()

        self._thread = threading.Thread(target=self._poll_loop, name="seller-world-poll", daemon=True)
        self._thread.start()

        log.info(
            f"SellerWorld started: polling hash='{SELLABLE_RESOURCES_HASH}' "
            f"every {SELLER_POLL_INTERVAL_SECONDS}s, {len(self._sellers)} seller(s) known at startup"
        )
        if not self._sellers:
            log.warning(
                f"No sellers known yet -- hash '{SELLABLE_RESOURCES_HASH}' is empty or no seller has "
                f"published. Every demand will be rejected until at least one seller publishes."
            )

    # ---- public interface (mirrors the old SimulatedWorld / stream version) ----
    @property
    def sellers(self):
        with self._lock:
            return list(self._sellers.keys())

    @property
    def capacity(self):
        """dict keyed (seller_id, resource) -> currently-available capacity, same shape as before."""
        with self._lock:
            return {
                (sid, r): info["capacity"].get(r, 0.0)
                for sid, info in self._sellers.items()
                for r in RESOURCES
            }

    @property
    def carbon(self):
        with self._lock:
            return {sid: info["carbon"] for sid, info in self._sellers.items()}

    def initial_remaining(self):
        """A fresh snapshot -- callers mutate their own private copy."""
        return dict(self.capacity)

    def snapshot(self):
        """
        {seller_id: {"capacity": {...}, "carbon": float, "last_seen": ts}}
        -- what algorithms.common.resync_remaining() diffs against to
        decide which sellers have fresher telemetry than a processor has
        already incorporated.
        """
        with self._lock:
            return {
                sid: {
                    "capacity": dict(info["capacity"]),
                    "carbon": info["carbon"],
                    "last_seen": info["last_seen"],
                    "collected_at": info.get("collected_at"),
                }
                for sid, info in self._sellers.items()
            }

    def latency_for_buyer(self, buyer_id: str) -> dict:
        """
        No real buyer<->seller network/geo model yet -- every known
        seller gets a flat DEFAULT_LATENCY unless it published a location,
        in which case _estimate_latency() is the one place to plug in a
        real distance/RTT estimate later.
        """
        with self._lock:
            snapshot = {sid: info for sid, info in self._sellers.items()}
        return {sid: self._estimate_latency(buyer_id, info) for sid, info in snapshot.items()}

    def _estimate_latency(self, buyer_id, info):
        if info.get("location") is not None:
            # Placeholder hook: once buyers also report a location, replace
            # this with a real haversine/RTT estimate. For now, having a
            # location at least distinguishes "we could estimate this" in
            # logs/telemetry from "no location data at all".
            return DEFAULT_LATENCY
        return DEFAULT_LATENCY

    def max_carbon(self):
        carbon = self.carbon
        return max(carbon.values()) if carbon else DEFAULT_CARBON

    def max_latency_estimate(self):
        return DEFAULT_LATENCY

    def stop(self):
        self._stop = True

    # ---- hash polling ----
    def _refresh_once(self):
        """Synchronous first read at construction time -- no catch-up needed, a HASH read is already the full current state."""
        try:
            raw = self._r.hgetall(SELLABLE_RESOURCES_HASH)
        except Exception as exc:
            log.error(f"Initial HGETALL '{SELLABLE_RESOURCES_HASH}' failed: {exc}", exc_info=True)
            raw = {}
        self._apply_snapshot(raw)

    def _poll_loop(self):
        while not self._stop:
            time.sleep(SELLER_POLL_INTERVAL_SECONDS)
            if self._stop:
                break
            try:
                raw = self._r.hgetall(SELLABLE_RESOURCES_HASH)
            except Exception as exc:
                log.error(
                    f"HGETALL '{SELLABLE_RESOURCES_HASH}' failed, retrying in "
                    f"{SELLER_POLL_INTERVAL_SECONDS}s: {exc}", exc_info=True,
                )
                continue
            self._apply_snapshot(raw)

    def _apply_snapshot(self, raw: dict):
        """
        raw is the full {field_key: json_value} dict from one HGETALL.
        Replaces our view of every seller present in it, tracking
        last_seen as "last time this seller's raw value actually
        changed" rather than "last time we saw it in the hash" -- see
        the module docstring's staleness note for why.
        """
        now = time.time()
        seen_ids = set()
        new_sellers = []
        updated_sellers = []

        with self._lock:
            for field_key, raw_value in raw.items():
                parsed = _parse_seller_entry(field_key, raw_value)
                if parsed is None:
                    log.warning(f"Skipping malformed entry in '{SELLABLE_RESOURCES_HASH}' field='{field_key}': {raw_value!r}")
                    continue

                seller_id = parsed["seller_id"]
                seen_ids.add(seller_id)
                existing = self._sellers.get(seller_id)
                changed = existing is None or existing.get("_raw") != raw_value
                last_seen = now if changed else existing["last_seen"]

                self._sellers[seller_id] = {
                    "capacity": parsed["capacity"],
                    "carbon": parsed["carbon"],
                    "location": parsed["location"],
                    "region": parsed["region"],
                    "collected_at": parsed.get("collected_at"),
                    "last_seen": last_seen,
                    "_raw": raw_value,
                }

                if existing is None:
                    new_sellers.append((seller_id, parsed))
                elif changed:
                    updated_sellers.append((seller_id, parsed))

            vanished = [sid for sid in self._sellers if sid not in seen_ids]
            for sid in vanished:
                del self._sellers[sid]

        for seller_id, parsed in new_sellers:
            log.info(
                f"NEW seller discovered: {seller_id} capacity={parsed['capacity']} "
                f"carbon={parsed['carbon']} region={parsed['region']} "
                f"collected_at={parsed.get('collected_at')}"
            )
        for seller_id, parsed in updated_sellers:
            log.debug(
                f"Updated seller: {seller_id} capacity={parsed['capacity']} "
                f"collected_at={parsed.get('collected_at')}"
            )
        for sid in vanished:
            log.warning(
                f"Seller {sid} no longer present in '{SELLABLE_RESOURCES_HASH}' -- "
                f"removed from the active pool (treated as offline)"
            )

        self._expire_stale()

    def _expire_stale(self):
        if not SELLER_STALE_AFTER_SECONDS:
            return
        now = time.time()
        with self._lock:
            stale = [
                sid for sid, info in self._sellers.items()
                if now - info["last_seen"] > SELLER_STALE_AFTER_SECONDS
            ]
            for sid in stale:
                del self._sellers[sid]
        for sid in stale:
            log.warning(
                f"Seller {sid}'s reported capacity hasn't changed in >{SELLER_STALE_AFTER_SECONDS}s -- "
                f"removed from the active pool (treated as offline)"
            )


_world_instance = None
_world_lock = threading.Lock()


def get_world() -> SellerWorld:
    """Process-local singleton -- one SellerWorld (and one polling thread) per process."""
    global _world_instance
    with _world_lock:
        if _world_instance is None:
            _world_instance = SellerWorld()
        return _world_instance

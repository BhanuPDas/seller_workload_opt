"""
Live seller world -- replaces the deterministic SimulatedWorld with a
registry built by continuously tailing a Redis Stream that real sellers
publish their own state to.

Same public interface the old simulated version had (`sellers`,
`capacity`, `carbon`, `latency_for_buyer`, `initial_remaining`, plus a
new `snapshot()`), so nothing in algorithms/ or workers/ needed to
change for this swap -- this file (and a small resync hook added to each
algorithm processor -- see algorithms/common.py's resync_remaining())
was the only thing that had to change.

*** REAL SELLER MESSAGE SCHEMA ***
Sellers report their current state as this JSON document:

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

    node            required. The seller/node id. Anything else is
                    skipped + logged as malformed.
    collected_at    optional ISO8601 timestamp of when the seller took
                    this reading. Stored as metadata only -- staleness
                    (SELLER_STALE_AFTER_SECONDS) is judged by this
                    process's own local receipt time (last_seen), not
                    by collected_at, to avoid clock-skew between nodes.
    sellable        cpu/ram/GPU/storage -- CURRENTLY AVAILABLE capacity
                    for that resource (not total/max -- see the
                    capacity-authority note below). Field names are
                    matched case-insensitively and aliased onto our
                    internal resource names (ram -> mem, GPU -> gpu,
                    storage -> storage, cpu -> cpu); see
                    _RESOURCE_ALIASES. A missing resource defaults to 0,
                    meaning "this seller doesn't currently have any of
                    this resource free."

Redis Streams can only hold flat string->string fields, so this nested
document has to be flattened somehow before it reaches XADD. Since the
real seller->Redis bridge is a separate component, _parse_seller_message()
below is written to accept whichever of these shapes it turns out to use,
without needing another round of changes:

    Pattern A (most likely): the whole JSON document as the value of a
    single stream field, e.g. {"data": "<json string>"} or
    {"message": "<json string>"} -- detected by scanning field values
    for one that starts with "{" and contains "node".
    Pattern B: dotted-flattened keys, e.g. {"node": "...",
    "collected_at": "...", "sellable.cpu": "238.0", "sellable.ram": "..."}.
    Pattern C: bare flattened keys, e.g. {"node": "...", "cpu": "238.0",
    "ram": "...", "GPU": "...", "storage": "..."} alongside "node" at the
    top level.

If the real bridge settles on one specific shape, the other two branches
in _parse_seller_message() are dead code but harmless -- no need to strip
them out.

*** CAPACITY AUTHORITY: "seller's next update overwrites ours" ***
Every fresh message for a seller REPLACES this registry's number for
that seller outright (last-write-wins, no averaging/merging). Each
algorithm processor keeps its own private `remaining` dict that it
still decrements locally between refreshes (via
algorithms.common.resync_remaining(), called at the top of every
decision), so a single run still shows a consistent, depleting picture
within one algorithm's own sequence of decisions -- but the moment a
seller publishes a fresh number, that seller's capacity snaps back to
whatever was just reported, discarding whatever we'd locally guessed.
Nothing here ever writes back to the seller side, so this is a read-only
subscriber.

If your real sellers instead publish TOTAL/max capacity (not currently-
available), you have two options: (a) have whatever bridges real
sellers to this stream compute and publish "available = total - in_use"
itself, which is the cleanest fix and keeps this file unchanged, or (b)
change what this file treats as ground truth by tracking total capacity
separately and letting only OUR OWN commits (never the seller's report)
reduce `remaining` -- ask if you want that version instead, since it's
a materially different reconciliation policy than the one implemented
here.
"""
import json
import threading
import time

from seller_workload_opt.v3.common.config import (
    SELLER_STREAM, SELLER_STALE_AFTER_SECONDS, DEFAULT_CARBON, DEFAULT_LATENCY,
    RESOURCES,
)
from seller_workload_opt.v3.common.logging_setup import get_logger
from seller_workload_opt.v3.common.redis_client import get_redis

log = get_logger("world")

_CATCH_UP_PAGE_SIZE = 500

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


def _extract_capacity(source: dict, prefix: str = "") -> dict:
    """
    Scans a flat dict for keys that map (after stripping an optional
    prefix like "sellable.") onto one of our resource names, case-
    insensitively. Returns a full {r: 0.0, ...} dict for every entry in
    RESOURCES so callers never have to guard for a missing key.
    """
    capacity = {r: 0.0 for r in RESOURCES}
    for k, v in source.items():
        key = k[len(prefix):] if prefix and k.startswith(prefix) else k
        mapped = _resource_key(key)
        if mapped is None or mapped not in capacity:
            continue
        try:
            capacity[mapped] = float(v)
        except (TypeError, ValueError):
            pass
    return capacity


def _find_embedded_json(fields: dict):
    """
    Pattern A: the whole seller document was JSON-encoded into a single
    stream field's value (common when a nested dict has to cross a
    flat-fields-only Redis Stream). Detected by scanning values for one
    that parses as a JSON object containing "node".
    """
    for v in fields.values():
        if not isinstance(v, str):
            continue
        stripped = v.strip()
        if not stripped.startswith("{") or '"node"' not in stripped:
            continue
        try:
            doc = json.loads(stripped)
        except (TypeError, ValueError):
            continue
        if isinstance(doc, dict) and "node" in doc:
            return doc
    return None


def _parse_seller_message(fields: dict):
    """
    fields is the flat str->str dict as read off one Redis Stream entry
    (or, for Pattern A, a dict already unwrapped from an embedded JSON
    document -- see _find_embedded_json). Returns
    {seller_id, capacity, carbon, location, region, collected_at} or
    None if the message is unusable (no node/seller id) -- caller logs
    and skips it.
    """
    # Pattern A: whole document embedded as one field's JSON string. The
    # embedded doc is the source of truth for node/collected_at/sellable,
    # but carbon/region/lat/lon aren't part of the real seller schema --
    # if a bridge ever adds them, it'll do so as sibling top-level stream
    # fields alongside the embedded-JSON field, not inside it. So keep
    # falling back to the original outer `fields` for those, rather than
    # discarding it once an embedded doc is found.
    outer_fields = fields
    embedded = _find_embedded_json(fields)
    if embedded is not None:
        fields = embedded

    seller_id = fields.get("node") or fields.get("seller_id") or fields.get("id")
    if not seller_id:
        return None

    sellable = fields.get("sellable")
    if isinstance(sellable, dict):
        # Already-nested dict (Pattern A after JSON parsing).
        capacity = _extract_capacity(sellable)
    else:
        # Pattern B (dotted-flattened, e.g. "sellable.cpu") or Pattern C
        # (bare flattened keys alongside "node"). _extract_capacity
        # strips the "sellable." prefix when present and otherwise just
        # matches bare resource key names, so one pass handles both.
        capacity = _extract_capacity(fields, prefix="sellable.")

    # carbon/region/lat/lon: prefer the embedded doc's copy if present
    # (e.g. a bridge that adds them into the JSON itself), else fall back
    # to sibling top-level stream fields, since the real schema doesn't
    # define where these would live.
    raw_carbon = fields.get("carbon", outer_fields.get("carbon"))
    try:
        carbon = float(raw_carbon) if raw_carbon not in (None, "") else DEFAULT_CARBON
    except (TypeError, ValueError):
        carbon = DEFAULT_CARBON

    raw_lat = fields.get("lat", outer_fields.get("lat"))
    raw_lon = fields.get("lon", outer_fields.get("lon"))
    location = None
    if raw_lat not in (None, "") and raw_lon not in (None, ""):
        try:
            location = (float(raw_lat), float(raw_lon))
        except (TypeError, ValueError):
            location = None
    region = fields.get("region") or outer_fields.get("region") or None
    collected_at = fields.get("collected_at") or None

    return {
        "seller_id": seller_id, "capacity": capacity, "carbon": carbon,
        "location": location, "region": region, "collected_at": collected_at,
    }


class SellerWorld:
    """
    Live registry of sellers built by tailing SELLER_STREAM. One instance
    per process (API and each of the 4 workers each run their own, via
    get_world() below) -- every process independently replays the full
    stream, since this is reference data everyone needs completely, not
    a work queue to divide up.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._sellers = {}   # seller_id -> {capacity, carbon, location, region, last_seen}
        self._r = get_redis()
        self._last_id = "0"
        self._stop = False

        self._catch_up()

        self._thread = threading.Thread(target=self._tail_loop, name="seller-world-tail", daemon=True)
        self._thread.start()

        log.info(
            f"SellerWorld started: tailing stream='{SELLER_STREAM}', "
            f"{len(self._sellers)} seller(s) known at startup"
        )
        if not self._sellers:
            log.warning(
                f"No sellers known yet -- stream '{SELLER_STREAM}' is empty or no seller has "
                f"published. Every demand will be rejected until at least one seller publishes."
            )

    # ---- public interface (mirrors the old SimulatedWorld) ----
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

    # ---- stream tailing ----
    def _catch_up(self):
        """
        Read the full stream once at startup (XRANGE, not XREAD) so we
        know every seller's last-reported state immediately, including
        sellers that published before this process started and haven't
        published again since.

        NOTE: cost scales with total historical message count on
        SELLER_STREAM, not with the number of distinct sellers. If
        whatever publishes to this stream doesn't already trim it
        (XADD ... MAXLEN ~ N), a long-running deployment will make this
        catch-up slower over time for any newly-started process -- worth
        trimming upstream since we only ever care about the latest
        message per seller.
        """
        last_id = "0"
        total = 0
        while True:
            range_min = f"({last_id}" if last_id != "0" else "-"
            entries = self._r.xrange(SELLER_STREAM, min=range_min, max="+", count=_CATCH_UP_PAGE_SIZE)
            if not entries:
                break
            for entry_id, fields in entries:
                self._apply(fields)
                total += 1
                last_id = entry_id
            if len(entries) < _CATCH_UP_PAGE_SIZE:
                break
        self._last_id = last_id
        log.info(f"Catch-up complete: replayed {total} historical seller message(s), last_id={self._last_id}")

    def _tail_loop(self):
        while not self._stop:
            try:
                resp = self._r.xread({SELLER_STREAM: self._last_id}, block=5000, count=100)
            except Exception as exc:
                log.error(f"seller stream xread failed, retrying in 2s: {exc}", exc_info=True)
                time.sleep(2)
                continue

            if resp:
                for _, entries in resp:
                    for entry_id, fields in entries:
                        self._apply(fields)
                        self._last_id = entry_id

            self._expire_stale()

    def _apply(self, fields):
        parsed = _parse_seller_message(fields)
        if parsed is None:
            log.warning(f"Skipping malformed seller message (no node/seller_id): {fields}")
            return

        seller_id = parsed["seller_id"]
        with self._lock:
            is_new = seller_id not in self._sellers
            self._sellers[seller_id] = {
                "capacity": parsed["capacity"],
                "carbon": parsed["carbon"],
                "location": parsed["location"],
                "region": parsed["region"],
                "collected_at": parsed.get("collected_at"),
                "last_seen": time.time(),
            }

        if is_new:
            log.info(
                f"NEW seller discovered: {seller_id} capacity={parsed['capacity']} "
                f"carbon={parsed['carbon']} region={parsed['region']} "
                f"collected_at={parsed.get('collected_at')}"
            )
        else:
            log.debug(
                f"Updated seller: {seller_id} capacity={parsed['capacity']} "
                f"collected_at={parsed.get('collected_at')}"
            )

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
                f"Seller {sid} hasn't published in >{SELLER_STALE_AFTER_SECONDS}s -- "
                f"removed from the active pool (treated as offline)"
            )


_world_instance = None
_world_lock = threading.Lock()


def get_world() -> SellerWorld:
    """Process-local singleton -- one SellerWorld (and one tailing thread) per process."""
    global _world_instance
    with _world_lock:
        if _world_instance is None:
            _world_instance = SellerWorld()
        return _world_instance

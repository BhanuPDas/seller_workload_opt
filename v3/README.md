# Marketplace Allocator — 4-algorithm shadow-comparison harness

Implements the design discussed: a single ingestion API pushes every
buyer demand request onto one Redis Stream, and all 4 algorithms
(RollingMILP+, RollingMILPPred, BatchMILP+, PrimalDual) consume the
identical stream independently, each with its own private capacity
ledger, so you can compare what each one would have decided on the
exact same input. A comparator service joins the 4 results per demand
item and logs which one is "most optimal."

**Nothing here commits real seller capacity.** Seller capacity/carbon
comes from a live Redis Stream that real sellers publish their own
state to (`common/world.py`'s `SellerWorld`) — see **Live seller data**
below for the message shape and the capacity-reconciliation policy.
Every other module only talks to the world through its interface
(`sellers`, `capacity`, `carbon`, `latency_for_buyer`, `snapshot`), so
that data source can change again later without touching the algorithm
or worker code — that's exactly what just happened going from the
original simulated capacity to this live-seller version.

## Why 4 independent workers instead of 1 shared allocator

RollingMILP+/RollingMILPPred/BatchMILP+ jointly optimize a *window* or
*batch* of items together — the seller assigned to item X depends on
which other items were in its window. So "run the same request through
all 4" can't mean 4 synchronous function calls returning at once; three
of the four only decide once enough items have buffered. Each worker
keeps its own private `remaining` capacity dict, reconciled against the
same live seller telemetry (see below), so the four are compared fairly
against identical seller state rather than reacting to each other's
allocations.

## Live seller data

Sellers publish their own state to a Redis Stream (`SELLER_STREAM`,
default `seller-updates`) — plain `XADD`, no consumer group, since every
process that needs seller state (the API + all 4 workers) independently
tails the full stream: this is reference data everyone needs a complete
copy of, not a work queue to divide up. See `common/world.py`'s module
docstring for the exhaustive version of everything below; the parsing
lives entirely in `_parse_seller_message()` there, which is the one
function to edit if your real message shape differs.

**Real seller message shape** — each seller reports:

```json
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
```

| field | required | meaning |
|---|---|---|
| `node` | yes | the seller id; anything else is skipped + logged as malformed |
| `collected_at` | no | ISO8601 timestamp of the seller's own reading; stored as metadata only — staleness is judged by this process's local receipt time (`last_seen`/`SELLER_STALE_AFTER_SECONDS`), not `collected_at`, to avoid clock skew across nodes |
| `sellable.cpu`, `sellable.ram`, `sellable.GPU`, `sellable.storage` | no (default 0) | **currently available** capacity for that resource — not total/max, see below. Matched case-insensitively and aliased onto our internal names (`ram`→`mem`, `GPU`→`gpu`, `storage` stays `storage`) |
| `carbon` | no (default `DEFAULT_CARBON`) | not part of the real seller schema (sellers don't report this) — same carbon-cost term used in the objective today, accepted at the top level if a bridge ever adds it |
| `region`, `lat`, `lon` | no | not used for anything real yet — see latency note below |

`storage` is tracked as a genuine 4th allocatable resource dimension
(`RESOURCES = ["cpu", "mem", "gpu", "storage"]` in `common/config.py`) —
buyers can request it via `resources.storage.demand_per_unit` and every
algorithm's capacity constraints and cost terms cover it exactly like
cpu/mem/gpu (all four loop generically over `RESOURCES`).

**Wire format on the stream**: Redis Streams can only hold flat
string→string fields, so the nested JSON above has to be flattened
somehow before it reaches `XADD`. `_parse_seller_message()` in
`common/world.py` accepts whichever of three shapes the real
seller→Redis bridge uses, without needing further changes: the whole
document JSON-encoded into a single field's value (what
`scripts/publish_fake_seller.py` does, and the most likely real shape),
dotted-flattened keys (`sellable.cpu`), or bare flattened keys (`cpu`,
`ram`, `GPU`, `storage` alongside `node` at the top level).

**Capacity authority: "seller's next update overwrites ours."** Every
fresh message for a seller replaces this registry's number for that
seller outright. Each algorithm keeps decrementing its own private
`remaining` between refreshes (via `resync_remaining()`, called at the
top of every decision) so a run still shows a consistent, depleting
picture within one algorithm's own sequence of hypothetical decisions —
but the moment a seller publishes again, that seller's capacity snaps to
whatever was just reported, discarding whatever we'd locally guessed.
This keeps feasibility checks grounded in what's actually free right
now, which matters because this is still a non-committing shadow
harness — nothing here ever reduces a real seller's actual capacity.

If your real sellers instead publish **total** capacity rather than
currently-available, either have whatever bridges them to this stream
compute and publish `available = total - in_use` (cleanest, keeps this
file unchanged), or say so and I'll wire up the other reconciliation
policy — tracking total separately and letting only *our own* commits
reduce `remaining`, never the seller's report. That's a materially
different policy from what's implemented now, not a small tweak.

**Latency** has no real network/geo model wired up yet — every seller
gets a flat `DEFAULT_LATENCY` regardless of `region`/`lat`/`lon`, until
there's a real distance/RTT source to plug into
`SellerWorld._estimate_latency()`.

**Sellers are fully dynamic now**: a new `node` is picked up the
moment it first publishes (no restart needed — verified by publishing a
new seller mid-run and watching it show up in the very next decision),
and a seller that hasn't published in `SELLER_STALE_AFTER_SECONDS`
(default 180s) is dropped from the active pool and logged as offline.

**Testing without real sellers connected yet**: `scripts/publish_fake_seller.py`
publishes synthetic seller telemetry in the exact expected shape —
either a one-shot batch (`--count 48`) or a looping heartbeat
(`--loop --interval 30`, useful so sellers don't get expired as stale
mid-test). `docker compose --profile test up seed-sellers` runs this
automatically as its own service (not started by `docker compose up`
alone — remove that service once real sellers are wired up).

## Architecture

```
buyer --POST /submit_demand--> API --XADD--> demand-stream (Redis Stream)
                                                   |
                    -------------------------------------------------------
                    |                |                |                  |
             cg-rolling_milp  cg-rolling_milp_pred  cg-batch_milp   cg-primal_dual
                    |                |                |                  |
              RollingMILP+ worker  RollingMILPPred worker  BatchMILP+ worker  PrimalDual worker
                    |                |                |                  |
                    -------------------------------------------------------
                                          |
                                 decisions-stream (Redis Stream)
                                          |
                                     comparator
                                          |
                          comparison:{demand_id}  (Redis HASH)
```

Each algorithm gets its own Redis Streams **consumer group** on the same
`demand-stream`, which is exactly the fan-out primitive you already use
elsewhere (consumer groups, not Pub/Sub) — every group receives every
message independently, in the same order, and tracks its own delivery
cursor and pending-entries list.

## Running it

```bash
docker compose up --build
```

This starts: `redis`, `api` (port 8080), the 4 workers, and `comparator`
— but with no real sellers connected yet, everything will reject until
at least one seller publishes to `seller-updates` (see **Live seller
data** below). For local testing before that's wired up:

```bash
docker compose --profile test up --build seed-sellers
```

Check `curl http://localhost:8080/sellers` to confirm sellers are known
before chasing "everything gets rejected" through the algorithm logs.

Watch any single algorithm's behavior in isolation with, e.g.:

```bash
docker compose logs -f worker-rolling-milp
docker compose logs -f comparator
```

Submit a demand request directly:

```bash
curl -X POST http://localhost:8080/submit_demand \
  -H "Content-Type: application/json" \
  -d '{
    "buyer_id": "B1",
    "app_type": "batch-inference",
    "ip": "10.0.0.5",
    "lease_duration": 3600,
    "resources": {
      "cpu": {"demand_per_unit": 8},
      "mem": {"demand_per_unit": 16},
      "gpu": {"demand_per_unit": 1},
      "storage": {"demand_per_unit": 50}
    }
  }'
```

The response includes `demand_id` and a `poll_url`. Poll it:

```bash
curl http://localhost:8080/decision/<demand_id>
```

`reported` lists which algorithms have already committed a decision for
this item; `pending` lists which are still buffering it in an open
window/batch (normal for RollingMILP+/RollingMILPPred/BatchMILP+ until
enough items accumulate — see tuning below). `comparison` appears once
all 4 have reported, and names the recommended algorithm/seller.

### Load-testing all 4 at once

```bash
pip install requests
python scripts/send_test_demand.py --num-buyers 5 --num-items 60 --interval 0.05
```

This submits a stream of randomized demand (same shape as the original
simulation's `generate_demands()`) and polls until each item has a full
4-way comparison, printing a summary. With the default `S`/`BATCH_SIZE`
(5 / 10), you'll see PrimalDual results almost immediately and the MILP
methods' results arrive in bursts as their windows/batches close.

## Reading the logs

Every worker logs, per algorithm:

- every arrival as it's buffered (`buffer_size=X/Y`, or immediate decision for PrimalDual)
- every solve: `SOLVING window_id=... members=[...]` before, `SOLVED window_id=... status=... solve_ms=...` after
- every commit or rejection, with the full cost breakdown
- a `window_id`/`batch_id` DONE summary line with running totals

This is deliberately close to the original simulation's `log`/`log_solve`
style, just persisted per-service via Docker logs instead of one shared
stdout. Every decision is also written to Redis (`decision:{demand_id}`
hash) tagged with `window_members` — the **full** window/batch
membership, not just the committed item — so you can reconstruct exactly
what else was competing for capacity when debugging a specific decision:

```bash
docker compose exec redis redis-cli HGETALL decision:<demand_id>
docker compose exec redis redis-cli HGETALL comparison:<demand_id>
```

The comparator logs a `COMPARISON` line per completed item and a
`WARNING` for any item that's been waiting more than
`COMPARATOR_WAIT_TIMEOUT_SECONDS` (default 60s) for a slow window/batch
to close — useful for spotting a stuck or under-filled window during
testing.

## Tuning

All the knobs from the simulation are environment variables (see
`common/config.py` for the full list and defaults) — set them per
service in `docker-compose.yml`, e.g. to shrink RollingMILP+'s window
for lower-latency testing:

```yaml
worker-rolling-milp:
  environment:
    <<: *common-env
    ROLLING_K: "10"
    ROLLING_S: "3"
    ROLLING_MAX_WAIT_SECONDS: "5"
```

`*_MAX_WAIT_SECONDS` bounds how long a window/batch can sit open waiting
to fill during low/sparse traffic — without it, RollingMILP+ or
BatchMILP+ could stall indefinitely if buyers arrive slower than
`S`/`BATCH_SIZE` items at a time.

## Known limitations (by design, for this testing phase)

- **A worker restart loses its cumulative capacity ledger, not just its in-flight buffer.** Crash recovery is verified end-to-end for *undelivered/uncommitted* items: if a worker dies mid-window, `XAUTOCLAIM` reclaims whatever was buffered-but-not-yet-committed on startup and re-feeds it (tested above by hard-killing `worker-rolling-milp` mid-window). What is **not** implemented is rebuilding `remaining` capacity from *already-committed* history — a fresh process starts from `world.initial_remaining()` again, so a restart effectively "forgets" every allocation the crashed instance had already committed. For a short-lived comparison run this is harmless; before this runs unattended for any length of time, add the event-sourcing replay mentioned in the earlier design discussion (append every commit to a durable log, replay it — or a periodic snapshot — on startup).
- **Single active instance per algorithm.** Each algorithm's consumer group assumes one live consumer owning that algorithm's `remaining` dict; running two replicas of the same worker would double-allocate capacity, same reasoning as the original simulation's single-threaded loop.
- **No real latency model yet** — see "Live seller data" above; every seller currently scores the same flat latency.
- **Seller catch-up cost scales with total historical messages on `SELLER_STREAM`, not seller count.** If nothing trims that stream (`XADD ... MAXLEN ~ N` on the producer side), a newly-started process's startup catch-up gets slower over time even though the number of distinct sellers stays the same.

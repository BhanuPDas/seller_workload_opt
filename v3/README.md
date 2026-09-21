# Marketplace Allocator — 4-algorithm shadow-comparison harness

Implements the design discussed: a single ingestion API pushes every
buyer demand request onto one Redis Stream, and all 4 algorithms
(RollingMILP+, RollingMILPPred, BatchMILP+, PrimalDual) consume the
identical stream independently, each with its own private simulated
seller-capacity ledger, so you can compare what each one would have
decided on the exact same input. A comparator service joins the 4
results per demand item and logs which one is "most optimal."

**Nothing here commits real seller capacity.** Per your instructions,
seller capacity/carbon/topology is a simulated, deterministic stand-in
(`common/world.py`) — the one module you'll swap out when this is wired
to real seller discovery. Every other module only talks to the world
through that module's interface (`sellers`, `capacity`, `carbon`,
`latency_for_buyer`), so that swap shouldn't require touching the
algorithm or worker code.

## Why 4 independent workers instead of 1 shared allocator

RollingMILP+/RollingMILPPred/BatchMILP+ jointly optimize a *window* or
*batch* of items together — the seller assigned to item X depends on
which other items were in its window. So "run the same request through
all 4" can't mean 4 synchronous function calls returning at once; three
of the four only decide once enough items have buffered. Each worker
keeps its own private `remaining` capacity dict, seeded identically
(same `WORLD_SEED` across all services), so the four are compared fairly
on identical starting conditions rather than reacting to each other's
allocations.

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

This starts: `redis`, `api` (port 8080), the 4 workers, and `comparator`.
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
      "gpu": {"demand_per_unit": 1}
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
- **Simulated world only** — see the section below for what to swap.

## What to change when real sellers replace the simulation

Everything funnels through `common/world.py`'s `SimulatedWorld` class.
Replace `get_world()` with something backed by real seller
discovery/capacity/topology behind the same 4 methods
(`sellers`, `capacity`, `carbon`, `latency_for_buyer`) and the API,
algorithms, and workers don't need to change. The one thing to preserve:
whatever replaces it must give an identical view to all 4 workers at any
given moment for the comparison to stay meaningful — if you move to real
sellers you'll need to decide whether the 4 workers read a shared live
capacity snapshot (introduces the concurrency questions we discussed
separately) or continue to run against their own independent replicas
for pure comparison purposes.

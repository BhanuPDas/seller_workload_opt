"""
Marketplace allocation simulation — v2

Changes vs the original script, and why (see the accompanying writeup for
full empirical evidence). Each change below was validated by running the
affected method against GlobalMILP over several seeds before/after.

1. SEEDING IS NOW ON BY DEFAULT (BASE_SEED + run).
   Reproducibility for development; see NUM_MACRO_SEEDS below for how to
   still validate that conclusions aren't an artifact of one seed family.

2. RollingMILP+ default lookahead widened: K=6,S=2 -> K=20,S=5.
   This is essentially free (CBC solve time barely changes at this problem
   size — 20 sellers, K<=20 items) and measurably closes the gap to
   GlobalMILP (in testing, avg latency gap vs Global shrank by ~40-60%
   across several seeds, with 0 extra rejections).

3. RollingMILPPred rewritten (rolling_milp_pred -> now uses a SOFT cost
   penalty for predicted near-future demand instead of subtracting it
   directly from remaining capacity). The original hard-reservation version
   was manufacturing extra rejections (1.5-2% reject rate vs 0% for the
   non-predictive RollingMILP+) that had nothing to do with real
   contention: a small forecast was enough to tip an already-tight seller
   into infeasibility for that batch. The soft version keeps the
   predictive signal (steer away from sellers a buyer will likely need
   again soon) without ever making capacity infeasible on its own. In
   testing this brought RollingMILPPred's reject rate down to match
   RollingMILP+'s (~0%) while keeping latency competitive.

4. SmartGreedyV2 (smart_greedy_v3 -> smart_greedy_v4) hardened: replaced
   the unbounded, equally-weighted running mean of ALL past latencies with
   an EMA, and clamped the resulting ratio to [0,1] before it feeds
   alpha_eff / gamma_eff. In the original, gamma_eff = GAMMA_SCARCITY * (1
   - avg_lat/max_latency) can go NEGATIVE if avg_lat ever exceeds
   max_latency, which would flip the utilization penalty into a reward for
   overpacking. That was never observed to trigger in ~150 test runs here
   (ratio stayed <0.4), so it turned out NOT to be the dominant cause of
   SmartGreedyV2's high run-to-run variance — but it's still a latent bug
   worth closing since nothing prevents it from triggering on a harder
   instance (more buyers, sparser topology, longer streams). The dominant
   cause of the variance, confirmed empirically (see writeup), is that
   SmartGreedyV2's outcome is highly correlated with GlobalMILP's outcome
   run-to-run (r=0.98) — i.e. it's mostly the RANDOM SCENARIO (topology
   draw) that's noisy, not the algorithm. v4 is a strict hardening, not a
   variance fix.

Everything else (GlobalMILP, SimpleGreedy, BatchMILP+, PricingV2,
PrimalDual, PrimalDualPred, PrimalDualPredHybrid) is unchanged from the
original — those held up fine under testing.

-----------------------------------------------------------------------

Same algorithmic content as v2 (widened RollingMILP+ lookahead, soft-penalty
RollingMILPPred, hardened SmartGreedyV4 — see the long comment block near
the top for why), PLUS verbose print-based debug logging throughout:

  - every MILP solve (GlobalMILP, each RollingMILP+/BatchMILP+/RollingMILPPred
    window or batch) prints its status, solve time, and how many items in
    that window/batch got rejected
  - every online method (SimpleGreedy, SmartGreedyV4, PricingV2, PrimalDual*)
    prints periodic progress (every ONLINE_LOG_EVERY stream items) plus a
    final summary line (rejects, elapsed time)
  - every run prints its seed and, per method, elapsed time + the resulting
    stats (avg latency / CO2 / rejection rate)

Toggle verbosity with the flags right below the imports. Everything is
plain print() to stdout -- no logging module, no files.
"""

import time
import networkx as nx
import numpy as np
import pulp as pl
from tabulate import tabulate

# ---------------------------------------------------------------------------
# Debug logging -- plain print statements, toggle-able
# ---------------------------------------------------------------------------
VERBOSE = True          # per-run / per-method progress + summaries
VERBOSE_SOLVES = True   # per MILP solve call (window/batch level). This is
                         # the noisiest flag -- RollingMILP+ alone prints
                         # ~12 lines/run at the default K=20,S=5, so 50 runs
                         # -> ~600 lines just for that method. Set False for
                         # a quieter console on big NUM_RUNS sweeps.
VERBOSE_ONLINE = True   # periodic progress print inside the non-MILP methods
ONLINE_LOG_EVERY = 20   # print online-method progress every N stream items

_t_start = time.time()


def log(msg, level="INFO"):
    """Elapsed-time-stamped console line. Respects VERBOSE."""
    if not VERBOSE:
        return
    elapsed = time.time() - _t_start
    print(f"[{elapsed:8.2f}s][{level:5s}] {msg}")


def log_solve(msg):
    if VERBOSE_SOLVES:
        log(msg, level="SOLVE")


def log_trace(msg):
    if VERBOSE_ONLINE:
        log(msg, level="TRACE")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
# BASE_SEED fixes one *family* of NUM_RUNS random scenarios so runs are
# reproducible while you iterate on the algorithms. It does NOT mean "the"
# canonical benchmark result -- different BASE_SEED values still give
# somewhat different NUM_RUNS-run averages. NUM_MACRO_SEEDS lets you repeat
# the whole experiment across several independent seed families in one go,
# so you can see directly whether a ranking/conclusion holds up or is a
# seed artifact, without having to manually edit BASE_SEED and rerun.
#
# Rule of thumb: NUM_MACRO_SEEDS=1 while you're iterating on code (fast,
# reproducible, lets you compare "before" vs "after" on the identical
# scenarios). Bump to 3+ once before you write down a conclusion like
# "RollingMILP+ beats PrimalDual" -- if it holds across 3 independent seed
# families you can trust it; if the ranking flips between families, you
# don't have enough runs per family yet (raise NUM_RUNS) or the two
# methods are genuinely close and the "winner" isn't well-defined.
BASE_SEED = 44
NUM_MACRO_SEEDS = 1     # bump to 3 (e.g.) to sanity-check stability across seed families
NUM_RUNS = 50           # scenarios per macro seed

RESOURCES = ["cpu", "mem", "gpu"]

ALPHA = 1.0
BETA = 0.3
GAMMA = {"cpu": 10, "mem": 2, "gpu": 20}
GAMMA_SCARCITY = {"cpu": 0.5, "mem": 0.15, "gpu": 1.0}

REJECTION_PENALTY = 100


def _binval(var):
    v = var.value()
    return 0.0 if v is None else v


def _solved_ok(model):
    return pl.LpStatus[model.status] == "Optimal"


# =====================================================
# TOPOLOGY
# =====================================================
def build_topology(num_buyers=5, num_sellers=20, buyer_degree=5):
    buyers = [f"B{i}" for i in range(num_buyers)]
    sellers = [f"S{i}" for i in range(num_sellers)]

    G = nx.Graph()
    degree = min(buyer_degree, len(sellers))

    for b in buyers:
        connected_sellers = np.random.choice(sellers, size=degree, replace=False)
        for s in connected_sellers:
            G.add_edge(b, s, weight=np.random.randint(1, 21))

    for i in range(num_sellers - 1):
        G.add_edge(sellers[i], sellers[i + 1], weight=np.random.randint(1, 10))

    L = {}
    for b in buyers:
        lengths = nx.single_source_dijkstra_path_length(G, b, weight="weight")
        for s in sellers:
            L[(b, s)] = lengths.get(s, 50)

    log(f"build_topology: {num_buyers} buyers, {num_sellers} sellers, "
        f"max_latency={max(L.values())}, min_latency={min(L.values())}")
    return buyers, sellers, L


# =====================================================
# DATA
# =====================================================
def generate_capacities(sellers):
    capacity, carbon = {}, {}
    for s in sellers:
        capacity[(s, "cpu")] = np.random.randint(32, 129)
        capacity[(s, "mem")] = np.random.randint(128, 513)
        capacity[(s, "gpu")] = np.random.randint(1, 9)
        carbon[s] = np.random.uniform(1, 10)
    total_cpu = sum(capacity[(s, "cpu")] for s in sellers)
    log(f"generate_capacities: total_cpu_capacity={total_cpu}, "
        f"avg_carbon={np.mean(list(carbon.values())):.2f}")
    return capacity, carbon


def generate_demands(buyers, max_workloads=60):
    demands = {b: [] for b in buyers}
    for t in range(max_workloads):
        b = np.random.choice(buyers)
        demands[b].append({
            "cpu": np.random.randint(2, 31),
            "mem": np.random.randint(4, 33),
            "gpu": np.random.randint(0, 3),
            "_arrival": t,
        })
    counts = {b: len(demands[b]) for b in buyers}
    log(f"generate_demands: {max_workloads} total items, per-buyer counts={counts}")
    return demands


def flatten_demands(demands):
    flat = [(b, d_idx, d) for b in demands for d_idx, d in enumerate(demands[b])]
    flat.sort(key=lambda x: x[2]["_arrival"])
    return flat


# =====================================================
# GLOBAL MILP (WITH REJECTION) — oracle upper bound
# =====================================================
def solve_milp(buyers, sellers, demands, capacity, carbon, L):
    model = pl.LpProblem("GlobalMILP", pl.LpMinimize)

    y = {(b, d_idx, s): pl.LpVariable(f"y_{b}_{d_idx}_{s}", cat="Binary")
         for b in buyers for d_idx in range(len(demands[b])) for s in sellers}
    z = {(b, d_idx): pl.LpVariable(f"z_{b}_{d_idx}", cat="Binary")
         for b in buyers for d_idx in range(len(demands[b]))}

    for b in buyers:
        for d_idx in range(len(demands[b])):
            model += pl.lpSum(y[(b, d_idx, s)] for s in sellers) == z[(b, d_idx)]

    for s in sellers:
        for r in RESOURCES:
            model += pl.lpSum(
                demands[b][d_idx][r] * y[(b, d_idx, s)]
                for b in buyers for d_idx in range(len(demands[b]))
            ) <= capacity[(s, r)]

    model += (
        pl.lpSum(
            (ALPHA * L[(b, s)] + BETA * carbon[s]) * y[(b, d_idx, s)]
            for b in buyers for d_idx in range(len(demands[b])) for s in sellers
        )
        + pl.lpSum(
            REJECTION_PENALTY * (1 - z[(b, d_idx)])
            for b in buyers for d_idx in range(len(demands[b]))
        )
    )

    n_items = sum(len(demands[b]) for b in buyers)
    log_solve(f"GlobalMILP: solving {n_items} items x {len(sellers)} sellers "
               f"({len(y)} binary y-vars, {len(z)} z-vars)...")
    t0 = time.time()
    model.solve(pl.PULP_CBC_CMD(timeLimit=60, msg=0))
    solve_time = time.time() - t0
    ok = _solved_ok(model)
    status_str = pl.LpStatus[model.status]
    obj_val = pl.value(model.objective) if ok else None
    log_solve(f"GlobalMILP: status={status_str} time={solve_time:.2f}s objective={obj_val}")
    if not ok:
        log(f"GlobalMILP: solver did NOT reach Optimal (status={status_str}) "
            f"within the 60s time limit -- treating unresolved items as rejected.",
            level="WARN")

    alloc = {}
    n_rejected = 0
    for b in buyers:
        for d_idx in range(len(demands[b])):
            if not ok or _binval(z[(b, d_idx)]) < 0.5:
                alloc[(b, d_idx)] = None
                n_rejected += 1
            else:
                alloc[(b, d_idx)] = None
                for s in sellers:
                    if _binval(y[(b, d_idx, s)]) > 0.5:
                        alloc[(b, d_idx)] = s
                        break
    log_solve(f"GlobalMILP: done -- {n_rejected}/{n_items} rejected")
    return alloc


# =====================================================
# SIMPLE GREEDY
# =====================================================
def simple_greedy(buyers, sellers, demands, L, capacity):
    remaining = {(s, r): capacity[(s, r)] for s in sellers for r in RESOURCES}
    alloc = {}
    t0 = time.time()
    n_rejected = 0
    n_items = 0
    for b in buyers:
        for d_idx, demand in enumerate(demands[b]):
            n_items += 1
            for s in sorted(sellers, key=lambda s: L[(b, s)]):
                if all(demand[r] <= remaining[(s, r)] for r in RESOURCES):
                    alloc[(b, d_idx)] = s
                    for r in RESOURCES:
                        remaining[(s, r)] -= demand[r]
                    break
            else:
                alloc[(b, d_idx)] = None
                n_rejected += 1
    log(f"SimpleGreedy: done -- {n_rejected}/{n_items} rejected, time={time.time()-t0:.3f}s")
    return alloc


# =====================================================
# SMART GREEDY V4 (hardened) — EMA + clamped adaptive weights
# =====================================================
def smart_greedy_v4(sellers, stream, L, capacity, carbon, ema_alpha=0.1):
    remaining = {(s, r): capacity[(s, r)] for s in sellers for r in RESOURCES}
    alloc = {}

    max_latency = max(L.values())
    max_carbon = max(carbon.values())

    ema_lat_ratio = 0.0  # bounded proxy for "how congested/far have recent picks been"
    t0 = time.time()
    n_rejected = 0

    for idx, (b, d_idx, d) in enumerate(stream):

        r_clamped = min(max(ema_lat_ratio, 0.0), 1.0)
        alpha_eff = ALPHA * (1 + r_clamped)
        gamma_eff = {r: GAMMA_SCARCITY[r] * (1 - r_clamped) for r in RESOURCES}  # always >= 0

        best_s, best_cost = None, float("inf")

        for s in sellers:
            if all(d[r] <= remaining[(s, r)] for r in RESOURCES):
                lat = L[(b, s)] / max_latency
                co2 = carbon[s] / max_carbon

                util_penalty = 0
                for r in RESOURCES:
                    util = 1 - remaining[(s, r)] / capacity[(s, r)]
                    util_penalty += gamma_eff[r] * (util ** 2) * (d[r] / capacity[(s, r)])

                cost = alpha_eff * lat + BETA * co2 + util_penalty

                if cost < best_cost:
                    best_cost, best_s = cost, s

        if best_s:
            alloc[(b, d_idx)] = best_s
            for r in RESOURCES:
                remaining[(best_s, r)] -= d[r]
            chosen_ratio = L[(b, best_s)] / max_latency
            ema_lat_ratio = ema_alpha * chosen_ratio + (1 - ema_alpha) * ema_lat_ratio
        else:
            alloc[(b, d_idx)] = None
            n_rejected += 1

        if idx % ONLINE_LOG_EVERY == 0:
            log_trace(f"SmartGreedyV4[{idx}/{len(stream)}]: buyer={b} -> seller={best_s} "
                       f"ema_lat_ratio={ema_lat_ratio:.3f} alpha_eff={alpha_eff:.3f} "
                       f"gamma_eff_cpu={gamma_eff['cpu']:.3f}")

    log(f"SmartGreedyV4: done -- {n_rejected}/{len(stream)} rejected, "
        f"final ema_lat_ratio={ema_lat_ratio:.3f}, time={time.time()-t0:.3f}s")
    return alloc


# =====================================================
# ROLLING MILP+ (WITH REJECTION) — widened default lookahead
# =====================================================
def rolling_milp(sellers, stream, capacity, carbon, L, K=20, S=5):
    assert 1 <= S <= K, "S must satisfy 1 <= S <= K"

    remaining = {(s, r): capacity[(s, r)] for s in sellers for r in RESOURCES}
    alloc = {}

    i = 0
    window_idx = 0
    n_rejected = 0
    n_items = len(stream)
    t_total0 = time.time()

    while i < len(stream):
        window = stream[i:i + K]
        commit_n = min(S, len(window))

        model = pl.LpProblem("RollingMILP", pl.LpMinimize)

        y = {(j, s): pl.LpVariable(f"y_{j}_{s}", cat="Binary")
             for j in range(len(window)) for s in sellers}
        z = {j: pl.LpVariable(f"z_{j}", cat="Binary") for j in range(len(window))}

        for j in range(len(window)):
            model += pl.lpSum(y[(j, s)] for s in sellers) == z[j]

        for s in sellers:
            for r in RESOURCES:
                model += pl.lpSum(window[j][2][r] * y[(j, s)]
                                   for j in range(len(window))) <= remaining[(s, r)]

        model += (
            pl.lpSum(
                (ALPHA * L[(window[j][0], s)] + BETA * carbon[s]) * y[(j, s)]
                for j in range(len(window)) for s in sellers
            )
            + pl.lpSum(REJECTION_PENALTY * (1 - z[j]) for j in range(len(window)))
        )

        t0 = time.time()
        model.solve(pl.PULP_CBC_CMD(timeLimit=10, msg=0))
        solve_time = time.time() - t0
        ok = _solved_ok(model)
        status_str = pl.LpStatus[model.status]

        window_rejected = 0
        for j in range(commit_n):
            b, d_idx, d = window[j]
            if not ok or _binval(z[j]) < 0.5:
                alloc[(b, d_idx)] = None
                window_rejected += 1
            else:
                alloc[(b, d_idx)] = None
                for s in sellers:
                    if _binval(y[(j, s)]) > 0.5:
                        alloc[(b, d_idx)] = s
                        for r in RESOURCES:
                            remaining[(s, r)] -= d[r]
                        break

        n_rejected += window_rejected
        log_solve(f"RollingMILP+ window#{window_idx} items[{i}:{i+len(window)}] "
                   f"(size={len(window)}, committing={commit_n}): status={status_str} "
                   f"time={solve_time:.3f}s rejected_in_window={window_rejected}")

        i += commit_n
        window_idx += 1

    log(f"RollingMILP+: done -- {n_rejected}/{n_items} rejected across "
        f"{window_idx} windows, total_time={time.time()-t_total0:.2f}s")
    return alloc


# =====================================================
# BATCH MILP (WITH REJECTION)
# =====================================================
def batch_milp(sellers, stream, capacity, carbon, L, BATCH_SIZE=10):
    remaining = {(s, r): capacity[(s, r)] for s in sellers for r in RESOURCES}
    alloc = {}

    batch_idx = 0
    n_rejected = 0
    n_items = len(stream)
    t_total0 = time.time()

    for i in range(0, len(stream), BATCH_SIZE):
        batch = stream[i:i + BATCH_SIZE]

        model = pl.LpProblem("BatchMILP", pl.LpMinimize)

        y = {(j, s): pl.LpVariable(f"y_{j}_{s}", cat="Binary")
             for j in range(len(batch)) for s in sellers}
        z = {j: pl.LpVariable(f"z_{j}", cat="Binary") for j in range(len(batch))}

        for j in range(len(batch)):
            model += pl.lpSum(y[(j, s)] for s in sellers) == z[j]

        for s in sellers:
            for r in RESOURCES:
                model += pl.lpSum(batch[j][2][r] * y[(j, s)]
                                   for j in range(len(batch))) <= remaining[(s, r)]

        model += (
            pl.lpSum(
                (ALPHA * L[(batch[j][0], s)] + BETA * carbon[s]) * y[(j, s)]
                for j in range(len(batch)) for s in sellers
            )
            + pl.lpSum(REJECTION_PENALTY * (1 - z[j]) for j in range(len(batch)))
        )

        t0 = time.time()
        model.solve(pl.PULP_CBC_CMD(timeLimit=10, msg=0))
        solve_time = time.time() - t0
        ok = _solved_ok(model)
        status_str = pl.LpStatus[model.status]

        batch_rejected = 0
        for j, (b, d_idx, d) in enumerate(batch):
            if not ok or _binval(z[j]) < 0.5:
                alloc[(b, d_idx)] = None
                batch_rejected += 1
            else:
                alloc[(b, d_idx)] = None
                for s in sellers:
                    if _binval(y[(j, s)]) > 0.5:
                        alloc[(b, d_idx)] = s
                        for r in RESOURCES:
                            remaining[(s, r)] -= d[r]
                        break

        n_rejected += batch_rejected
        log_solve(f"BatchMILP+ batch#{batch_idx} items[{i}:{i+len(batch)}] "
                   f"(size={len(batch)}): status={status_str} time={solve_time:.3f}s "
                   f"rejected_in_batch={batch_rejected}")
        batch_idx += 1

    log(f"BatchMILP+: done -- {n_rejected}/{n_items} rejected across "
        f"{batch_idx} batches, total_time={time.time()-t_total0:.2f}s")
    return alloc


# =====================================================
# PRICING V3 (STABLE)
# =====================================================
def pricing_v3(sellers, stream, capacity, carbon, L):
    remaining = {(s, r): capacity[(s, r)] for s in sellers for r in RESOURCES}
    prices = {(s, r): 0.0 for s in sellers for r in RESOURCES}
    alloc = {}
    t0 = time.time()
    n_rejected = 0

    for idx, (b, d_idx, d) in enumerate(stream):
        best_s, best_cost = None, float("inf")

        for s in sellers:
            if all(d[r] <= remaining[(s, r)] for r in RESOURCES):
                price_cost = sum(prices[(s, r)] * (d[r] / capacity[(s, r)]) for r in RESOURCES)
                cost = ALPHA * L[(b, s)] + BETA * carbon[s] + price_cost
                if cost < best_cost:
                    best_cost, best_s = cost, s

        if best_s:
            alloc[(b, d_idx)] = best_s
            for r in RESOURCES:
                remaining[(best_s, r)] -= d[r]
                util = 1 - remaining[(best_s, r)] / capacity[(best_s, r)]
                prices[(best_s, r)] = 0.8 * prices[(best_s, r)] + 0.2 * (util * GAMMA[r])
        else:
            alloc[(b, d_idx)] = None
            n_rejected += 1

        if idx % ONLINE_LOG_EVERY == 0:
            avg_price = np.mean(list(prices.values()))
            log_trace(f"PricingV2[{idx}/{len(stream)}]: buyer={b} -> seller={best_s} "
                       f"avg_price={avg_price:.3f}")

    log(f"PricingV2: done -- {n_rejected}/{len(stream)} rejected, time={time.time()-t0:.3f}s")
    return alloc


def primal_dual_online(sellers, stream, capacity, carbon, L,
                        eta=0.5, decay=0.01, gamma_s=2.0):
    remaining = {(s, r): capacity[(s, r)] for s in sellers for r in RESOURCES}
    lam = {(s, r): 0.0 for s in sellers for r in RESOURCES}
    alloc = {}
    t0 = time.time()
    n_rejected = 0

    for idx, (b, d_idx, d) in enumerate(stream):
        best_s, best_cost = None, float("inf")

        for s in sellers:
            if all(d[r] <= remaining[(s, r)] for r in RESOURCES):
                price_cost = sum(lam[(s, r)] * (d[r] / capacity[(s, r)]) for r in RESOURCES)
                scarcity = 0.0
                for r in RESOURCES:
                    rem = remaining[(s, r)]
                    if rem > 0:
                        scarcity += (d[r] / rem) ** 2
                cost = ALPHA * L[(b, s)] + BETA * carbon[s] + price_cost + gamma_s * scarcity
                if cost < best_cost:
                    best_cost, best_s = cost, s

        if best_s is not None:
            alloc[(b, d_idx)] = best_s
            for r in RESOURCES:
                remaining[(best_s, r)] -= d[r]
                lam[(best_s, r)] += eta * (d[r] / capacity[(best_s, r)])
                lam[(best_s, r)] *= (1 - decay)
        else:
            alloc[(b, d_idx)] = None
            n_rejected += 1

        if idx % ONLINE_LOG_EVERY == 0:
            avg_lam = np.mean(list(lam.values()))
            log_trace(f"PrimalDual[{idx}/{len(stream)}]: buyer={b} -> seller={best_s} "
                       f"avg_lambda={avg_lam:.3f}")

    log(f"PrimalDual: done -- {n_rejected}/{len(stream)} rejected, time={time.time()-t0:.3f}s")
    return alloc


# =====================================================
# ROLLING MILP PRED v2 (soft predictive penalty, replaces the old hard
# capacity-reservation version — see v2's module docstring for why)
# =====================================================
def rolling_milp_pred(sellers, stream, capacity, carbon, L, K=20, S=5,
                       W=20, F=2, W_min=5, pred_weight=0.5, ema_alpha=0.5):
    remaining = {(s, r): capacity[(s, r)] for s in sellers for r in RESOURCES}
    alloc = {}

    buyers = set(b for b, _, _ in stream)
    ema_demand = {b: {r: 0.0 for r in RESOURCES} for b in buyers}
    history_count = {b: 0 for b in buyers}

    i = 0
    window_idx = 0
    n_rejected = 0
    n_items = len(stream)
    t_total0 = time.time()

    while i < len(stream):
        window = stream[i:i + K]
        commit_n = min(S, len(window))

        forecast = {}
        for b in buyers:
            conf = min(history_count[b] / W, 1.0) if history_count[b] >= W_min else 0.0
            forecast[b] = {r: ema_demand[b][r] * F * conf for r in RESOURCES}

        model = pl.LpProblem("RollingMILP_Pred_v2", pl.LpMinimize)

        y = {(j, s): pl.LpVariable(f"y_{j}_{s}", cat="Binary")
             for j in range(len(window)) for s in sellers}
        z = {j: pl.LpVariable(f"z_{j}", cat="Binary") for j in range(len(window))}

        for j in range(len(window)):
            model += pl.lpSum(y[(j, s)] for s in sellers) == z[j]

        for s in sellers:
            for r in RESOURCES:
                model += pl.lpSum(window[j][2][r] * y[(j, s)]
                                   for j in range(len(window))) <= remaining[(s, r)]

        pred_cost = pl.lpSum(
            pred_weight * sum(forecast[window[j][0]][r] / capacity[(s, r)] for r in RESOURCES) * y[(j, s)]
            for j in range(len(window)) for s in sellers
        )

        model += (
            pl.lpSum(
                (ALPHA * L[(window[j][0], s)] + BETA * carbon[s]) * y[(j, s)]
                for j in range(len(window)) for s in sellers
            )
            + pred_cost
            + pl.lpSum(REJECTION_PENALTY * (1 - z[j]) for j in range(len(window)))
        )

        t0 = time.time()
        model.solve(pl.PULP_CBC_CMD(timeLimit=10, msg=0))
        solve_time = time.time() - t0
        ok = _solved_ok(model)
        status_str = pl.LpStatus[model.status]

        window_rejected = 0
        for j in range(commit_n):
            b, d_idx, d = window[j]
            if not ok or _binval(z[j]) < 0.5:
                alloc[(b, d_idx)] = None
                window_rejected += 1
            else:
                alloc[(b, d_idx)] = None
                for s in sellers:
                    if _binval(y[(j, s)]) > 0.5:
                        alloc[(b, d_idx)] = s
                        for r in RESOURCES:
                            remaining[(s, r)] -= d[r]
                        break
            for r in RESOURCES:
                ema_demand[b][r] = ema_alpha * d[r] + (1 - ema_alpha) * ema_demand[b][r]
            history_count[b] += 1

        n_rejected += window_rejected
        max_forecast_cpu = max((forecast[b]["cpu"] for b in buyers), default=0)
        log_solve(f"RollingMILPPred window#{window_idx} items[{i}:{i+len(window)}] "
                   f"(size={len(window)}, committing={commit_n}): status={status_str} "
                   f"time={solve_time:.3f}s rejected_in_window={window_rejected} "
                   f"max_forecast_cpu={max_forecast_cpu:.2f}")

        i += commit_n
        window_idx += 1

    log(f"RollingMILPPred: done -- {n_rejected}/{n_items} rejected across "
        f"{window_idx} windows, total_time={time.time()-t_total0:.2f}s")
    return alloc


# =====================================================
# PREDICTIVE PRIMAL DUAL (v2)
# =====================================================
def primal_dual_online_pred(sellers, stream, capacity, carbon, L,
                             eta=0.5, decay=0.01, gamma_s=2.0,
                             pred_alpha=0.5, pred_window=10):
    remaining = {(s, r): capacity[(s, r)] for s in sellers for r in RESOURCES}
    lam = {(s, r): 0.0 for s in sellers for r in RESOURCES}
    alloc = {}
    t0 = time.time()
    n_rejected = 0

    ema_demand = {b: {r: 0.0 for r in RESOURCES} for b, _, _ in stream}
    history_count = {b: 0 for b, _, _ in stream}

    for idx, (b, d_idx, d) in enumerate(stream):
        best_s, best_cost = None, float("inf")

        for s in sellers:
            if all(d[r] <= remaining[(s, r)] for r in RESOURCES):
                price_cost = sum(lam[(s, r)] * (d[r] / capacity[(s, r)]) for r in RESOURCES)
                scarcity = 0.0
                for r in RESOURCES:
                    rem = remaining[(s, r)]
                    if rem > 0:
                        pred_demand = ema_demand[b][r] if history_count[b] > 0 else 0.0
                        scarcity += ((d[r] + pred_demand) / rem) ** 2
                cost = ALPHA * L[(b, s)] + BETA * carbon[s] + price_cost + gamma_s * scarcity
                if cost < best_cost:
                    best_cost, best_s = cost, s

        if best_s is not None:
            alloc[(b, d_idx)] = best_s
            for r in RESOURCES:
                remaining[(best_s, r)] -= d[r]
                lam[(best_s, r)] += eta * (d[r] / capacity[(best_s, r)])
                lam[(best_s, r)] *= (1 - decay)
        else:
            alloc[(b, d_idx)] = None
            n_rejected += 1

        for r in RESOURCES:
            ema_demand[b][r] = pred_alpha * d[r] + (1 - pred_alpha) * ema_demand[b][r]
        history_count[b] += 1

        if idx % ONLINE_LOG_EVERY == 0:
            log_trace(f"PrimalDualPred[{idx}/{len(stream)}]: buyer={b} -> seller={best_s} "
                       f"history_count={history_count[b]}")

    log(f"PrimalDualPred: done -- {n_rejected}/{len(stream)} rejected, time={time.time()-t0:.3f}s")
    return alloc


# =====================================================
# PREDICTIVE ONLINE PRIMAL DUAL HYBRID
# =====================================================
def primal_dual_online_pred_hybrid(sellers, stream, capacity, carbon, L,
                                    eta=0.5, decay=0.01, gamma_s=2.0,
                                    pred_alpha=0.5, pred_window=10):
    remaining = {(s, r): capacity[(s, r)] for s in sellers for r in RESOURCES}
    lam = {(s, r): 0.0 for s in sellers for r in RESOURCES}
    alloc = {}
    t0 = time.time()
    n_rejected = 0
    n_using_prediction = 0

    ema_demand = {b: {r: 0.0 for r in RESOURCES} for b, _, _ in stream}
    history_count = {b: 0 for b, _, _ in stream}

    for idx, (b, d_idx, d) in enumerate(stream):
        use_prediction = history_count[b] >= pred_window
        if use_prediction:
            n_using_prediction += 1
        best_s, best_cost = None, float("inf")

        for s in sellers:
            if all(d[r] <= remaining[(s, r)] for r in RESOURCES):
                price_cost = sum(lam[(s, r)] * (d[r] / capacity[(s, r)]) for r in RESOURCES)
                scarcity = 0.0
                for r in RESOURCES:
                    rem = remaining[(s, r)]
                    if rem > 0:
                        if use_prediction:
                            pred_demand = ema_demand[b][r]
                            scarcity += ((d[r] + pred_demand) / rem) ** 2
                        else:
                            scarcity += (d[r] / rem) ** 2
                cost = ALPHA * L[(b, s)] + BETA * carbon[s] + price_cost + gamma_s * scarcity
                if cost < best_cost:
                    best_cost, best_s = cost, s

        if best_s is not None:
            alloc[(b, d_idx)] = best_s
            for r in RESOURCES:
                remaining[(best_s, r)] -= d[r]
                lam[(best_s, r)] += eta * (d[r] / capacity[(best_s, r)])
                lam[(best_s, r)] *= (1 - decay)
        else:
            alloc[(b, d_idx)] = None
            n_rejected += 1

        for r in RESOURCES:
            ema_demand[b][r] = pred_alpha * d[r] + (1 - pred_alpha) * ema_demand[b][r]
        history_count[b] += 1

        if idx % ONLINE_LOG_EVERY == 0:
            log_trace(f"PrimalDualPredHybrid[{idx}/{len(stream)}]: buyer={b} -> "
                       f"seller={best_s} use_prediction={use_prediction}")

    log(f"PrimalDualPredHybrid: done -- {n_rejected}/{len(stream)} rejected "
        f"({n_using_prediction} items used the predictive branch), "
        f"time={time.time()-t0:.3f}s")
    return alloc


# =====================================================
# STATS
# =====================================================
def compute_stats(alloc, demands, carbon, L):
    lat, co2 = [], []
    rejected = 0
    for (b, d_idx), s in alloc.items():
        if s:
            lat.append(L[(b, s)])
            co2.append(carbon[s])
        else:
            rejected += 1
    total = len(alloc)
    avg_lat = np.mean(lat) if lat else 0
    avg_co2 = np.mean(co2) if co2 else 0
    p95_lat = np.percentile(lat, 95) if lat else 0
    rejection_rate = rejected / total if total else 0.0
    return avg_lat, avg_co2, p95_lat, rejection_rate


def compute_utilization_metrics(alloc, demands, capacity, sellers):
    used = {(s, r): 0 for s in sellers for r in RESOURCES}
    for (b, d_idx), s in alloc.items():
        if s is None:
            continue
        for r in RESOURCES:
            used[(s, r)] += demands[b][d_idx][r]

    per_resource_util = {r: [] for r in RESOURCES}
    for s in sellers:
        for r in RESOURCES:
            cap = capacity[(s, r)]
            if cap > 0:
                per_resource_util[r].append(used[(s, r)] / cap)

    all_utils = [u for vals in per_resource_util.values() for u in vals]
    if not all_utils:
        return 0.0, 0.0

    variances = [np.var(vals) for vals in per_resource_util.values() if vals]
    util_var = np.mean(variances) if variances else 0.0
    max_util = np.max(all_utils)

    return util_var, max_util


# =====================================================
# MAIN
# =====================================================
def run_experiment(macro_seed):
    method_names = [
        "GlobalMILP", "SimpleGreedy", "SmartGreedyV4", "RollingMILP+",
        "BatchMILP+", "PricingV2", "PrimalDual", "PrimalDualPred",
        "RollingMILPPred", "PrimalDualPredHybrid",
    ]
    all_results = {m: {"avg_lat": [], "p95_lat": [], "avg_co2": [],
                        "util_var": [], "max_util": [], "rej": []}
                   for m in method_names}

    method_fns = {
        "GlobalMILP": lambda buyers, sellers, demands, stream, capacity, carbon, L:
            solve_milp(buyers, sellers, demands, capacity, carbon, L),
        "SimpleGreedy": lambda buyers, sellers, demands, stream, capacity, carbon, L:
            simple_greedy(buyers, sellers, demands, L, capacity),
        "SmartGreedyV4": lambda buyers, sellers, demands, stream, capacity, carbon, L:
            smart_greedy_v4(sellers, stream, L, capacity, carbon),
        "RollingMILP+": lambda buyers, sellers, demands, stream, capacity, carbon, L:
            rolling_milp(sellers, stream, capacity, carbon, L),
        "BatchMILP+": lambda buyers, sellers, demands, stream, capacity, carbon, L:
            batch_milp(sellers, stream, capacity, carbon, L),
        "PricingV2": lambda buyers, sellers, demands, stream, capacity, carbon, L:
            pricing_v3(sellers, stream, capacity, carbon, L),
        "PrimalDual": lambda buyers, sellers, demands, stream, capacity, carbon, L:
            primal_dual_online(sellers, stream, capacity, carbon, L),
        "PrimalDualPred": lambda buyers, sellers, demands, stream, capacity, carbon, L:
            primal_dual_online_pred(sellers, stream, capacity, carbon, L),
        "RollingMILPPred": lambda buyers, sellers, demands, stream, capacity, carbon, L:
            rolling_milp_pred(sellers, stream, capacity, carbon, L),
        "PrimalDualPredHybrid": lambda buyers, sellers, demands, stream, capacity, carbon, L:
            primal_dual_online_pred_hybrid(sellers, stream, capacity, carbon, L),
    }

    for run in range(NUM_RUNS):
        seed = macro_seed + run
        log(f"===== RUN {run+1}/{NUM_RUNS}  (seed={seed}) =====", level="RUN")
        np.random.seed(seed)

        buyers, sellers, L = build_topology()
        capacity, carbon = generate_capacities(sellers)
        demands = generate_demands(buyers)
        stream = flatten_demands(demands)

        for name in method_names:
            t0 = time.time()
            alloc = method_fns[name](buyers, sellers, demands, stream, capacity, carbon, L)
            elapsed = time.time() - t0

            avg_lat, avg_co2, p95_lat, rej = compute_stats(alloc, demands, carbon, L)
            util_var, max_util = compute_utilization_metrics(alloc, demands, capacity, sellers)

            log(f"RUN {run+1} / {name}: avg_lat={avg_lat:.3f} p95_lat={p95_lat:.3f} "
                f"avg_co2={avg_co2:.3f} util_var={util_var:.4f} max_util={max_util:.3f} "
                f"reject={rej:.4f} elapsed={elapsed:.2f}s", level="RESULT")

            all_results[name]["avg_lat"].append(avg_lat)
            all_results[name]["p95_lat"].append(p95_lat)
            all_results[name]["avg_co2"].append(avg_co2)
            all_results[name]["util_var"].append(util_var)
            all_results[name]["max_util"].append(max_util)
            all_results[name]["rej"].append(rej)

    return method_names, all_results


def main():
    log(f"Starting experiment: BASE_SEED={BASE_SEED} NUM_MACRO_SEEDS={NUM_MACRO_SEEDS} "
        f"NUM_RUNS={NUM_RUNS} (VERBOSE_SOLVES={VERBOSE_SOLVES}, "
        f"VERBOSE_ONLINE={VERBOSE_ONLINE})", level="CONFIG")

    macro_seeds = [BASE_SEED + 1000 * k for k in range(NUM_MACRO_SEEDS)]

    for ms in macro_seeds:
        log(f"### Macro seed family starting at {ms} ###", level="MACRO")
        t0 = time.time()
        method_names, all_results = run_experiment(ms)
        log(f"### Macro seed family {ms} done in {time.time()-t0:.1f}s ###", level="MACRO")

        final_results = []
        for name in method_names:
            final_results.append([
                name,
                np.mean(all_results[name]["avg_lat"]),
                np.mean(all_results[name]["p95_lat"]),
                np.mean(all_results[name]["avg_co2"]),
                np.mean(all_results[name]["util_var"]),
                np.mean(all_results[name]["max_util"]),
                np.mean(all_results[name]["rej"]),
            ])

        print(f"\n=== FINAL COMPARISON (macro_seed={ms}, averaged over {NUM_RUNS} runs) ===")
        print(tabulate(final_results,
                        headers=["Method", "Avg Lat", "P95 Lat", "Avg CO2",
                                 "Util Var", "Max Util", "Reject Rate"],
                        tablefmt="fancy_grid"))


if __name__ == "__main__":
    main()
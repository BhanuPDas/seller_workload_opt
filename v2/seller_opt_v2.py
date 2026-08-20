import networkx as nx
import numpy as np
import pulp as pl
from tabulate import tabulate

# Base seed for reproducibility — each run in main() seeds with BASE_SEED + run,
# so the whole 50-run experiment is reproducible end-to-end, and any single run
# can be regenerated/debugged in isolation (e.g. np.random.seed(BASE_SEED + 23))
# without needing to replay every prior run first.
BASE_SEED = 44

RESOURCES = ["cpu","mem","gpu"]

ALPHA = 1.0
BETA  = 0.3
GAMMA = {"cpu":10, "mem":2, "gpu":20}   # used by pricing_v3 (unnormalized latency scale there)

# smart_greedy_v3 works in a NORMALIZED [0,1] cost space (lat/max_latency,
# carbon/max_carbon), so it needs its own, much smaller scarcity weights —
# reusing raw GAMMA there let the utilization-penalty term (up to ~20-30)
# completely swamp the ~0-2 latency/carbon terms. This is scaled down so all
# three cost components land in comparable ranges.
GAMMA_SCARCITY = {"cpu": 0.5, "mem": 0.15, "gpu": 1.0}

REJECTION_PENALTY = 100


def _binval(var):
    """Safely read a PuLP binary variable's value, defaulting to 0.0 if the
    solver never assigned one (e.g. infeasible/aborted solve)."""
    v = var.value()
    return 0.0 if v is None else v


def _solved_ok(model):
    return pl.LpStatus[model.status] == "Optimal"


# =====================================================
# TOPOLOGY
# =====================================================
def build_topology(num_buyers=5, num_sellers=20, buyer_degree=5):
    buyers  = [f"B{i}" for i in range(num_buyers)]
    sellers = [f"S{i}" for i in range(num_sellers)]

    G = nx.Graph()

    # Guard against num_sellers < buyer_degree, which would otherwise crash
    # np.random.choice(..., replace=False).
    degree = min(buyer_degree, len(sellers))

    for b in buyers:
        connected_sellers = np.random.choice(sellers, size=degree, replace=False)
        for s in connected_sellers:
            G.add_edge(b, s, weight=np.random.randint(1,21))

    for i in range(num_sellers-1):
        G.add_edge(sellers[i], sellers[i+1], weight=np.random.randint(1,10))

    L = {}
    for b in buyers:
        lengths = nx.single_source_dijkstra_path_length(G, b, weight="weight")
        for s in sellers:
            L[(b,s)] = lengths.get(s, 50)

    return buyers, sellers, L


# =====================================================
# DATA
# =====================================================
def generate_capacities(sellers):
    capacity, carbon = {}, {}
    for s in sellers:
        capacity[(s,"cpu")] = np.random.randint(32,129)
        capacity[(s,"mem")] = np.random.randint(128,513)
        capacity[(s,"gpu")] = np.random.randint(1,9)
        carbon[s] = np.random.uniform(1,10)
    return capacity, carbon


def generate_demands(buyers, max_workloads=60):
    """Each demand is tagged with its true generation order (`_arrival`) so
    that flatten_demands can reconstruct the actual interleaved arrival
    sequence across buyers, instead of grouping everything by buyer."""
    demands = {b: [] for b in buyers}
    for t in range(max_workloads):
        b = np.random.choice(buyers)
        demands[b].append({
            "cpu": np.random.randint(2,31),
            "mem": np.random.randint(4,33),
            "gpu": np.random.randint(0,3),
            "_arrival": t,
        })
    return demands


def flatten_demands(demands):
    """Return the stream in true arrival order (interleaved across buyers),
    not grouped buyer-by-buyer. Grouping by buyer meant every batch/rolling
    window saw almost exclusively one buyer at a time, so the online/batch
    methods rarely faced real cross-buyer contention."""
    flat = [(b, d_idx, d)
            for b in demands
            for d_idx, d in enumerate(demands[b])]
    flat.sort(key=lambda x: x[2]["_arrival"])
    return flat


# =====================================================
# GLOBAL MILP (WITH REJECTION)
# =====================================================
def solve_milp(buyers, sellers, demands, capacity, carbon, L):
    model = pl.LpProblem("GlobalMILP", pl.LpMinimize)

    y = {(b,d_idx,s): pl.LpVariable(f"y_{b}_{d_idx}_{s}", cat="Binary")
         for b in buyers
         for d_idx in range(len(demands[b]))
         for s in sellers}

    z = {(b,d_idx): pl.LpVariable(f"z_{b}_{d_idx}", cat="Binary")
         for b in buyers
         for d_idx in range(len(demands[b]))}

    for b in buyers:
        for d_idx in range(len(demands[b])):
            model += pl.lpSum(y[(b,d_idx,s)] for s in sellers) == z[(b,d_idx)]

    for s in sellers:
        for r in RESOURCES:
            model += pl.lpSum(
                demands[b][d_idx][r] * y[(b,d_idx,s)]
                for b in buyers for d_idx in range(len(demands[b]))
            ) <= capacity[(s,r)]

    model += (
        pl.lpSum(
            (ALPHA * L[(b,s)] + BETA * carbon[s]) * y[(b,d_idx,s)]
            for b in buyers for d_idx in range(len(demands[b])) for s in sellers
        )
        +
        pl.lpSum(
            REJECTION_PENALTY * (1 - z[(b,d_idx)])
            for b in buyers for d_idx in range(len(demands[b]))
        )
    )

    model.solve(pl.PULP_CBC_CMD(timeLimit=60, msg=0))
    ok = _solved_ok(model)

    alloc = {}
    for b in buyers:
        for d_idx in range(len(demands[b])):
            if not ok or _binval(z[(b,d_idx)]) < 0.5:
                alloc[(b,d_idx)] = None
            else:
                alloc[(b,d_idx)] = None  # default if no y found (shouldn't happen when ok)
                for s in sellers:
                    if _binval(y[(b,d_idx,s)]) > 0.5:
                        alloc[(b,d_idx)] = s
                        break

    return alloc


# =====================================================
# SIMPLE GREEDY
# =====================================================
def simple_greedy(buyers, sellers, demands, L, capacity):
    remaining = {(s,r): capacity[(s,r)] for s in sellers for r in RESOURCES}
    alloc = {}

    for b in buyers:
        for d_idx, demand in enumerate(demands[b]):
            for s in sorted(sellers, key=lambda s: L[(b,s)]):
                if all(demand[r] <= remaining[(s,r)] for r in RESOURCES):
                    alloc[(b,d_idx)] = s
                    for r in RESOURCES:
                        remaining[(s,r)] -= demand[r]
                    break
            else:
                alloc[(b,d_idx)] = None

    return alloc


# =====================================================
# SMART GREEDY V3 (ADAPTIVE)
# =====================================================
def smart_greedy_v3(sellers, stream, L, capacity, carbon):
    remaining = {(s,r): capacity[(s,r)] for s in sellers for r in RESOURCES}
    alloc = {}

    max_latency = max(L.values())
    max_carbon = max(carbon.values())

    lat_history = []

    for (b,d_idx,d) in stream:

        avg_lat = np.mean(lat_history) if lat_history else 0

        alpha_eff = ALPHA * (1 + avg_lat / max_latency)
        gamma_eff = {
            r: GAMMA_SCARCITY[r] * (1 - avg_lat / max_latency)
            for r in RESOURCES
        }

        best_s, best_cost = None, float("inf")

        for s in sellers:
            if all(d[r] <= remaining[(s,r)] for r in RESOURCES):

                lat = L[(b,s)] / max_latency
                co2 = carbon[s] / max_carbon

                util_penalty = 0
                for r in RESOURCES:
                    util = 1 - remaining[(s,r)] / capacity[(s,r)]
                    util_penalty += gamma_eff[r] * (util**2) * (d[r] / capacity[(s,r)])

                cost = alpha_eff*lat + BETA*co2 + util_penalty

                if cost < best_cost:
                    best_cost, best_s = cost, s

        if best_s:
            alloc[(b,d_idx)] = best_s
            for r in RESOURCES:
                remaining[(best_s,r)] -= d[r]
            lat_history.append(L[(b,best_s)])
        else:
            alloc[(b,d_idx)] = None

    return alloc


# =====================================================
# ROLLING MILP (WITH REJECTION) — receding-horizon variant
# =====================================================
def rolling_milp(sellers, stream, capacity, carbon, L, K=6, S=2):
    """
    Rolling-horizon MILP, distinct from batch_milp by design:

      - Looks ahead over a window of K items for cross-buyer contention.
      - Solves the MILP jointly over that window.
      - Commits only the FIRST S decisions (S < K), then advances by S.

    Because S < K, every commit still benefits from (K - S) items of extra
    lookahead beyond what gets locked in — batch_milp has zero visibility
    past its own block boundary, and a naive "commit only item 0, slide by
    1" version re-solves once per item for no extra benefit. This sits in
    between: cheaper than solving per-item, more informed than pure batch.
    """
    assert 1 <= S <= K, "S must satisfy 1 <= S <= K"

    remaining = {(s,r): capacity[(s,r)] for s in sellers for r in RESOURCES}
    alloc = {}

    i = 0
    while i < len(stream):
        window = stream[i:i+K]
        commit_n = min(S, len(window))

        model = pl.LpProblem("RollingMILP", pl.LpMinimize)

        y = {(j,s): pl.LpVariable(f"y_{j}_{s}", cat="Binary")
             for j in range(len(window)) for s in sellers}

        z = {j: pl.LpVariable(f"z_{j}", cat="Binary")
             for j in range(len(window))}

        for j in range(len(window)):
            model += pl.lpSum(y[(j,s)] for s in sellers) == z[j]

        for s in sellers:
            for r in RESOURCES:
                model += pl.lpSum(window[j][2][r] * y[(j,s)]
                                  for j in range(len(window))) <= remaining[(s,r)]

        model += (
            pl.lpSum(
                (ALPHA*L[(window[j][0],s)] + BETA*carbon[s]) * y[(j,s)]
                for j in range(len(window)) for s in sellers
            )
            +
            pl.lpSum(REJECTION_PENALTY * (1 - z[j]) for j in range(len(window)))
        )

        model.solve(pl.PULP_CBC_CMD(timeLimit=10, msg=0))
        ok = _solved_ok(model)

        for j in range(commit_n):
            b, d_idx, d = window[j]
            if not ok or _binval(z[j]) < 0.5:
                alloc[(b,d_idx)] = None
            else:
                alloc[(b,d_idx)] = None
                for s in sellers:
                    if _binval(y[(j,s)]) > 0.5:
                        alloc[(b,d_idx)] = s
                        for r in RESOURCES:
                            remaining[(s,r)] -= d[r]
                        break

        i += commit_n

    return alloc


# =====================================================
# BATCH MILP (WITH REJECTION)
# =====================================================
def batch_milp(sellers, stream, capacity, carbon, L, BATCH_SIZE=10):
    remaining = {(s,r): capacity[(s,r)] for s in sellers for r in RESOURCES}
    alloc = {}

    for i in range(0, len(stream), BATCH_SIZE):
        batch = stream[i:i+BATCH_SIZE]

        model = pl.LpProblem("BatchMILP", pl.LpMinimize)

        y = {(j,s): pl.LpVariable(f"y_{j}_{s}", cat="Binary")
             for j in range(len(batch)) for s in sellers}

        z = {j: pl.LpVariable(f"z_{j}", cat="Binary")
             for j in range(len(batch))}

        for j in range(len(batch)):
            model += pl.lpSum(y[(j,s)] for s in sellers) == z[j]

        for s in sellers:
            for r in RESOURCES:
                model += pl.lpSum(batch[j][2][r] * y[(j,s)]
                                  for j in range(len(batch))) <= remaining[(s,r)]

        model += (
            pl.lpSum(
                (ALPHA*L[(batch[j][0],s)] + BETA*carbon[s]) * y[(j,s)]
                for j in range(len(batch)) for s in sellers
            )
            +
            pl.lpSum(REJECTION_PENALTY * (1 - z[j]) for j in range(len(batch)))
        )

        model.solve(pl.PULP_CBC_CMD(timeLimit=10, msg=0))
        ok = _solved_ok(model)

        for j,(b,d_idx,d) in enumerate(batch):
            if not ok or _binval(z[j]) < 0.5:
                alloc[(b,d_idx)] = None
            else:
                alloc[(b,d_idx)] = None
                for s in sellers:
                    if _binval(y[(j,s)]) > 0.5:
                        alloc[(b,d_idx)] = s
                        for r in RESOURCES:
                            remaining[(s,r)] -= d[r]
                        break

    return alloc


# =====================================================
# PRICING V3 (STABLE)
# =====================================================
def pricing_v3(sellers, stream, capacity, carbon, L):
    remaining = {(s,r): capacity[(s,r)] for s in sellers for r in RESOURCES}
    prices = {(s,r): 0.0 for s in sellers for r in RESOURCES}
    alloc = {}

    for (b,d_idx,d) in stream:
        best_s, best_cost = None, float("inf")

        for s in sellers:
            if all(d[r] <= remaining[(s,r)] for r in RESOURCES):

                price_cost = sum(
                    prices[(s,r)] * (d[r] / capacity[(s,r)])
                    for r in RESOURCES
                )

                cost = ALPHA*L[(b,s)] + BETA*carbon[s] + price_cost

                if cost < best_cost:
                    best_cost, best_s = cost, s

        if best_s:
            alloc[(b,d_idx)] = best_s

            for r in RESOURCES:
                remaining[(best_s,r)] -= d[r]

                util = 1 - remaining[(best_s,r)] / capacity[(best_s,r)]

                prices[(best_s,r)] = (
                    0.8 * prices[(best_s,r)]
                    + 0.2 * (util * GAMMA[r])
                )
        else:
            alloc[(b,d_idx)] = None

    return alloc


def primal_dual_online(sellers, stream, capacity, carbon, L,
                      eta=0.5, decay=0.01, gamma_s=2.0):
    """
    Online primal-dual with scarcity-aware penalty.

    eta      = dual step size
    decay    = price stabilization
    gamma_s  = strength of scarcity penalty
    """

    remaining = {(s, r): capacity[(s, r)] for s in sellers for r in RESOURCES}
    lam = {(s, r): 0.0 for s in sellers for r in RESOURCES}

    alloc = {}

    for (b, d_idx, d) in stream:

        best_s, best_cost = None, float("inf")

        for s in sellers:
            if all(d[r] <= remaining[(s, r)] for r in RESOURCES):

                price_cost = sum(
                    lam[(s, r)] * (d[r] / capacity[(s, r)])
                    for r in RESOURCES
                )

                scarcity = 0.0
                for r in RESOURCES:
                    rem = remaining[(s, r)]
                    if rem > 0:
                        scarcity += (d[r] / rem) ** 2

                cost = (
                    ALPHA * L[(b, s)] +
                    BETA * carbon[s] +
                    price_cost +
                    gamma_s * scarcity
                )

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

    return alloc


# =====================================================
# PREDICTIVE ROLLING MILP (v2)
# =====================================================
def rolling_milp_pred(sellers, stream, capacity, carbon, L, K=6,
                      W=20, F=2, W_min=5, alpha_pred=0.3, pred_cap_fraction=0.5, ema_alpha=0.5):
    """
    Rolling MILP with robust buyer-activeness prediction.

    Predictions are built from each buyer's EMA *before* the current
    demand is folded in (otherwise the "future" prediction partly consists
    of the very demand it's supposed to be predicting ahead of). W is used
    as the confidence ramp: with little history the reservation is scaled
    down; once a buyer has W observed demands, the full F-step-ahead
    reservation applies.
    """
    remaining = {(s,r): capacity[(s,r)] for s in sellers for r in RESOURCES}
    alloc = {}

    buyers = set(b for b,_,_ in stream)
    ema_demand = {b: {r: 0.0 for r in RESOURCES} for b in buyers}
    history_count = {b: 0 for b in buyers}

    i = 0
    while i < len(stream):
        batch = stream[i:i+K]

        batch_predicted_demand = []
        for (b,d_idx,d) in batch:
            prior_count = history_count[b]  # observations BEFORE this one

            if prior_count >= W_min:
                confidence = min(prior_count / W, 1.0)
                predicted_demand = {r: ema_demand[b][r] * F * confidence for r in RESOURCES}

                capped_demand = {}
                for r in RESOURCES:
                    max_cap = pred_cap_fraction * max(capacity[(s,r)] for s in sellers)
                    capped_demand[r] = min(predicted_demand[r] * alpha_pred, max_cap)

                batch_predicted_demand.append(capped_demand)
            else:
                batch_predicted_demand.append({r: 0 for r in RESOURCES})

            # Update EMA/history AFTER using it for prediction, so this
            # demand only ever informs *future* predictions, not its own.
            for r in RESOURCES:
                ema_demand[b][r] = ema_alpha * d[r] + (1 - ema_alpha) * ema_demand[b][r]
            history_count[b] += 1

        model = pl.LpProblem("RollingMILP_Pred_Stable", pl.LpMinimize)

        y = {(j,s): pl.LpVariable(f"y_{j}_{s}", cat="Binary") for j in range(len(batch)) for s in sellers}
        z = {j: pl.LpVariable(f"z_{j}", cat="Binary") for j in range(len(batch))}

        for j in range(len(batch)):
            model += pl.lpSum(y[(j,s)] for s in sellers) == z[j]

        for s in sellers:
            for r in RESOURCES:
                model += pl.lpSum(
                    (batch[j][2][r] + batch_predicted_demand[j][r]) * y[(j,s)]
                    for j in range(len(batch))
                ) <= remaining[(s,r)]

        model += (
            pl.lpSum(
                (ALPHA*L[(batch[j][0],s)] + BETA*carbon[s]) * y[(j,s)]
                for j in range(len(batch)) for s in sellers
            )
            +
            pl.lpSum(REJECTION_PENALTY * (1 - z[j]) for j in range(len(batch)))
        )

        model.solve(pl.PULP_CBC_CMD(timeLimit=10, msg=0))
        ok = _solved_ok(model)

        for j,(b,d_idx,d) in enumerate(batch):
            if not ok or _binval(z[j]) < 0.5:
                alloc[(b,d_idx)] = None
            else:
                alloc[(b,d_idx)] = None
                for s in sellers:
                    if _binval(y[(j,s)]) > 0.5:
                        alloc[(b,d_idx)] = s
                        for r in RESOURCES:
                            remaining[(s,r)] -= d[r]
                        break

        i += K

    return alloc


# =====================================================
# PREDICTIVE PRIMAL DUAL (v2)
# =====================================================
def primal_dual_online_pred(sellers, stream, capacity, carbon, L,
                      eta=0.5, decay=0.01, gamma_s=2.0,
                      pred_alpha=0.5, pred_window=10):
    """
    Online primal-dual with predictive EMA-based lookahead.

    The EMA used for prediction is always based on demands observed
    STRICTLY BEFORE the current one — it's updated with the current demand
    only after the allocation decision is made, so the model never "predicts"
    the very request it's currently deciding on. pred_window is treated as an
    exponential smoothing horizon (consistent with rolling_milp_pred's EMA)
    rather than a literal fixed-size buffer.
    """

    remaining = {(s, r): capacity[(s, r)] for s in sellers for r in RESOURCES}
    lam = {(s, r): 0.0 for s in sellers for r in RESOURCES}
    alloc = {}

    ema_demand = {b: {r: 0.0 for r in RESOURCES} for b, _, _ in stream}
    history_count = {b: 0 for b, _, _ in stream}

    for (b, d_idx, d) in stream:

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

        # Fold the current demand into the buyer's EMA AFTER the decision.
        for r in RESOURCES:
            ema_demand[b][r] = pred_alpha * d[r] + (1 - pred_alpha) * ema_demand[b][r]
        history_count[b] += 1

    return alloc


# =====================================================
# PREDICTIVE ONLINE PRIMAL DUAL HYBRID
# =====================================================
def primal_dual_online_pred_hybrid(sellers, stream, capacity, carbon, L,
                      eta=0.5, decay=0.01, gamma_s=2.0,
                      pred_alpha=0.5, pred_window=10):
    """
    Hybrid Primal-Dual:
    - Cold start: behaves exactly like PrimalDual
    - After warmup (pred_window PRIOR observations): switches to predictive
      scarcity using EMA. The warm-up check happens before this demand is
      folded into history, so "warmed up" genuinely means pred_window past
      observations, not pred_window-including-this-one.
    """

    remaining = {(s, r): capacity[(s, r)] for s in sellers for r in RESOURCES}
    lam = {(s, r): 0.0 for s in sellers for r in RESOURCES}
    alloc = {}

    ema_demand = {b: {r: 0.0 for r in RESOURCES} for b, _, _ in stream}
    history_count = {b: 0 for b, _, _ in stream}

    for (b, d_idx, d) in stream:

        use_prediction = history_count[b] >= pred_window

        best_s, best_cost = None, float("inf")

        for s in sellers:
            if all(d[r] <= remaining[(s, r)] for r in RESOURCES):

                price_cost = sum(
                    lam[(s, r)] * (d[r] / capacity[(s, r)])
                    for r in RESOURCES
                )

                scarcity = 0.0
                for r in RESOURCES:
                    rem = remaining[(s, r)]
                    if rem > 0:
                        if use_prediction:
                            pred_demand = ema_demand[b][r]
                            scarcity += ((d[r] + pred_demand) / rem) ** 2
                        else:
                            scarcity += (d[r] / rem) ** 2

                cost = (
                    ALPHA * L[(b, s)] +
                    BETA * carbon[s] +
                    price_cost +
                    gamma_s * scarcity
                )

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

        for r in RESOURCES:
            ema_demand[b][r] = pred_alpha * d[r] + (1 - pred_alpha) * ema_demand[b][r]
        history_count[b] += 1

    return alloc


# =====================================================
# STATS
# =====================================================
def compute_stats(alloc, demands, carbon, L):
    lat, co2 = [], []
    rejected = 0

    for (b,d_idx), s in alloc.items():
        if s:
            lat.append(L[(b,s)])
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
    used = {(s,r):0 for s in sellers for r in RESOURCES}

    for (b,d_idx), s in alloc.items():
        if s is None:
            continue
        for r in RESOURCES:
            used[(s,r)] += demands[b][d_idx][r]

    # CPU/mem/GPU utilization live on different scales, so pool them per
    # resource type first — otherwise "variance across (seller,resource)"
    # conflates cross-seller imbalance with cross-resource-type differences.
    per_resource_util = {r: [] for r in RESOURCES}
    for s in sellers:
        for r in RESOURCES:
            cap = capacity[(s,r)]
            if cap > 0:
                per_resource_util[r].append(used[(s,r)] / cap)

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
def main():

    NUM_RUNS = 50

    method_names = [
        "GlobalMILP",
        "SimpleGreedy",
        "SmartGreedyV2",
        "RollingMILP+",
        "BatchMILP+",
        "PricingV2",
        "PrimalDual",
        "PrimalDualPred",
        "RollingMILPPred",
        "PrimalDualPredHybrid"
    ]

    # store metrics across runs
    all_results = {
        m: {
            "avg_lat": [],
            "p95_lat": [],
            "avg_co2": [],
            "util_var": [],
            "max_util": [],
            "rej": []
        }
        for m in method_names
    }

    for run in range(NUM_RUNS):
        print(f"Run {run+1}/{NUM_RUNS}")

        # np.random.seed(BASE_SEED + run)

        buyers, sellers, L = build_topology()
        capacity, carbon = generate_capacities(sellers)
        demands = generate_demands(buyers)

        stream = flatten_demands(demands)

        methods = {
            "GlobalMILP": solve_milp(buyers, sellers, demands, capacity, carbon, L),
            "SimpleGreedy": simple_greedy(buyers, sellers, demands, L, capacity),
            "SmartGreedyV2": smart_greedy_v3(sellers, stream, L, capacity, carbon),
            "RollingMILP+": rolling_milp(sellers, stream, capacity, carbon, L),
            "BatchMILP+": batch_milp(sellers, stream, capacity, carbon, L),
            "PricingV2": pricing_v3(sellers, stream, capacity, carbon, L),
            "PrimalDual": primal_dual_online(sellers, stream, capacity, carbon, L),
            "PrimalDualPred": primal_dual_online_pred(sellers, stream, capacity, carbon, L),
            "RollingMILPPred": rolling_milp_pred(sellers, stream, capacity, carbon, L),
            "PrimalDualPredHybrid": primal_dual_online_pred_hybrid(sellers, stream, capacity, carbon, L)
        }

        for name, alloc in methods.items():
            avg_lat, avg_co2, p95_lat, rej = compute_stats(alloc, demands, carbon, L)
            util_var, max_util = compute_utilization_metrics(alloc, demands, capacity, sellers)

            all_results[name]["avg_lat"].append(avg_lat)
            all_results[name]["p95_lat"].append(p95_lat)
            all_results[name]["avg_co2"].append(avg_co2)
            all_results[name]["util_var"].append(util_var)
            all_results[name]["max_util"].append(max_util)
            all_results[name]["rej"].append(rej)

    # =========================
    # Aggregate results
    # =========================
    final_results = []

    for name in method_names:
        final_results.append([
            name,
            np.mean(all_results[name]["avg_lat"]),
            np.mean(all_results[name]["p95_lat"]),
            np.mean(all_results[name]["avg_co2"]),
            np.mean(all_results[name]["util_var"]),
            np.mean(all_results[name]["max_util"]),
            np.mean(all_results[name]["rej"])
        ])

    print("\n=== FINAL COMPARISON (AVERAGED OVER RUNS) ===")
    print(tabulate(final_results,
                   headers=[
                       "Method",
                       "Avg Lat",
                       "P95 Lat",
                       "Avg CO2",
                       "Util Var",
                       "Max Util",
                       "Reject Rate"
                   ],
                   tablefmt="fancy_grid"))

if __name__ == "__main__":
    main()
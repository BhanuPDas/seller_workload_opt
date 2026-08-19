import networkx as nx
import numpy as np
import pulp as pl
from tabulate import tabulate
from collections import deque

#np.random.seed(44)

RESOURCES = ["cpu","mem","gpu"]

ALPHA = 1.0
BETA  = 0.3
GAMMA = {"cpu":10, "mem":2, "gpu":20}

REJECTION_PENALTY = 100


# =====================================================
# TOPOLOGY
# =====================================================
def build_topology(num_buyers=5, num_sellers=20):
    buyers  = [f"B{i}" for i in range(num_buyers)]
    sellers = [f"S{i}" for i in range(num_sellers)]

    G = nx.Graph()

    for b in buyers:
        connected_sellers = np.random.choice(sellers, size=5, replace=False)
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
    demands = {b: [] for b in buyers}
    for _ in range(max_workloads):
        b = np.random.choice(buyers)
        demands[b].append({
            "cpu": np.random.randint(2,31),
            "mem": np.random.randint(4,33),
            "gpu": np.random.randint(0,3)
        })
    return demands


def flatten_demands(demands):
    return [(b, d_idx, d)
            for b in demands
            for d_idx, d in enumerate(demands[b])]


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

    model.solve(pl.PULP_CBC_CMD(timeLimit=60))

    alloc = {}
    for b in buyers:
        for d_idx in range(len(demands[b])):
            if z[(b,d_idx)].value() < 0.5:
                alloc[(b,d_idx)] = None
            else:
                for s in sellers:
                    if y[(b,d_idx,s)].value() > 0.5:
                        alloc[(b,d_idx)] = s

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
            r: GAMMA[r] * (1 - avg_lat / max_latency)
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
# ROLLING MILP (WITH REJECTION)
# =====================================================
def rolling_milp(sellers, stream, capacity, carbon, L, K=6):
    remaining = {(s,r): capacity[(s,r)] for s in sellers for r in RESOURCES}
    alloc = {}

    i = 0
    while i < len(stream):
        batch = stream[i:i+K]

        model = pl.LpProblem("RollingMILP", pl.LpMinimize)

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

        model.solve(pl.PULP_CBC_CMD(timeLimit=10))

        b,d_idx,d = batch[0]
        if z[0].value() < 0.5:
            alloc[(b,d_idx)] = None
        else:
            for s in sellers:
                if y[(0,s)].value() > 0.5:
                    alloc[(b,d_idx)] = s
                    for r in RESOURCES:
                        remaining[(s,r)] -= d[r]

        i += 1

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

        model.solve(pl.PULP_CBC_CMD(timeLimit=10))

        for j,(b,d_idx,d) in enumerate(batch):
            if z[j].value() < 0.5:
                alloc[(b,d_idx)] = None
            else:
                for s in sellers:
                    if y[(j,s)].value() > 0.5:
                        alloc[(b,d_idx)] = s
                        for r in RESOURCES:
                            remaining[(s,r)] -= d[r]

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

                # =========================
                # 1. Lagrangian price cost
                # =========================
                price_cost = sum(
                    lam[(s, r)] * (d[r] / capacity[(s, r)])
                    for r in RESOURCES
                )

                # =========================
                # 2. Scarcity penalty (NEW)
                # =========================
                scarcity = 0.0
                for r in RESOURCES:
                    rem = remaining[(s, r)]
                    if rem > 0:
                        scarcity += (d[r] / rem) ** 2

                # =========================
                # 3. Total cost
                # =========================
                cost = (
                    ALPHA * L[(b, s)] +
                    BETA * carbon[s] +
                    price_cost +
                    gamma_s * scarcity
                )

                if cost < best_cost:
                    best_cost, best_s = cost, s

        # =========================
        # APPLY DECISION
        # =========================
        if best_s is not None:
            alloc[(b, d_idx)] = best_s

            for r in RESOURCES:
                remaining[(best_s, r)] -= d[r]

                # Dual update
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
    Features:
      - EMA-based demand prediction
      - Confidence threshold to avoid sparse early predictions
      - Soft-lookahead: predicted demand added cautiously
    """
    remaining = {(s,r): capacity[(s,r)] for s in sellers for r in RESOURCES}
    alloc = {}

    # Track buyer history for EMA
    buyers = set(b for b,_,_ in stream)
    ema_demand = {b: {r: 0.0 for r in RESOURCES} for b in buyers}
    history_count = {b: 0 for b in buyers}  # number of observed demands

    i = 0
    while i < len(stream):
        batch = stream[i:i+K]

        batch_predicted_demand = []
        for (b,d_idx,d) in batch:
            # Update EMA
            history_count[b] += 1
            for r in RESOURCES:
                ema_demand[b][r] = ema_alpha * d[r] + (1 - ema_alpha) * ema_demand[b][r]

            # Only predict if enough history and low variance (confidence)
            if history_count[b] >= W_min:
                predicted_count = sum(1 for _ in range(history_count[b])) / history_count[b] * F
                predicted_demand = {r: ema_demand[b][r] * predicted_count for r in RESOURCES}

                # Blend with alpha_pred and cap
                capped_demand = {}
                for r in RESOURCES:
                    max_cap = pred_cap_fraction * max(capacity[(s,r)] for s in sellers)
                    capped_demand[r] = min(predicted_demand[r] * alpha_pred, max_cap)

                batch_predicted_demand.append(capped_demand)
            else:
                # Not enough history → no prediction
                batch_predicted_demand.append({r: 0 for r in RESOURCES})

        # Solve MILP for the batch
        model = pl.LpProblem("RollingMILP_Pred_Stable", pl.LpMinimize)

        y = {(j,s): pl.LpVariable(f"y_{j}_{s}", cat="Binary") for j in range(len(batch)) for s in sellers}
        z = {j: pl.LpVariable(f"z_{j}", cat="Binary") for j in range(len(batch))}

        # Assignment constraints
        for j in range(len(batch)):
            model += pl.lpSum(y[(j,s)] for s in sellers) == z[j]

        # Capacity constraints with predicted demand softly included
        for s in sellers:
            for r in RESOURCES:
                model += pl.lpSum(
                    (batch[j][2][r] + batch_predicted_demand[j][r]) * y[(j,s)]
                    for j in range(len(batch))
                ) <= remaining[(s,r)]

        # Objective: latency + carbon + rejection penalty
        model += (
            pl.lpSum(
                (ALPHA*L[(batch[j][0],s)] + BETA*carbon[s]) * y[(j,s)]
                for j in range(len(batch)) for s in sellers
            )
            +
            pl.lpSum(REJECTION_PENALTY * (1 - z[j]) for j in range(len(batch)))
        )

        model.solve(pl.PULP_CBC_CMD(timeLimit=10))

        # Apply allocations for batch
        for j,(b,d_idx,d) in enumerate(batch):
            if z[j].value() < 0.5:
                alloc[(b,d_idx)] = None
            else:
                for s in sellers:
                    if y[(j,s)].value() > 0.5:
                        alloc[(b,d_idx)] = s
                        for r in RESOURCES:
                            remaining[(s,r)] -= d[r]

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

    eta        = dual step size
    decay      = price stabilization
    gamma_s    = strength of scarcity penalty
    pred_alpha = EMA smoothing factor for prediction
    pred_window= window length to keep EMA per buyer
    """

    remaining = {(s, r): capacity[(s, r)] for s in sellers for r in RESOURCES}
    lam = {(s, r): 0.0 for s in sellers for r in RESOURCES}
    alloc = {}

    # EMA per buyer
    ema_demand = {b: {r: 0.0 for r in RESOURCES} for b, _, _ in stream}
    history = {b: [] for b, _, _ in stream}

    for (b, d_idx, d) in stream:

        # update EMA history
        history[b].append(d)
        if len(history[b]) > pred_window:
            history[b].pop(0)

        # compute EMA of past demands
        for r in RESOURCES:
            past_vals = [h[r] for h in history[b]]
            ema = past_vals[0]
            for val in past_vals[1:]:
                ema = pred_alpha*val + (1-pred_alpha)*ema
            ema_demand[b][r] = ema

        best_s, best_cost = None, float("inf")

        for s in sellers:
            if all(d[r] <= remaining[(s, r)] for r in RESOURCES):

                # 1. Lagrangian price cost
                price_cost = sum(lam[(s, r)] * (d[r] / capacity[(s, r)]) for r in RESOURCES)

                # 2. Scarcity penalty (actual + predicted)
                scarcity = 0.0
                for r in RESOURCES:
                    rem = remaining[(s, r)]
                    if rem > 0:
                        # include predicted demand in numerator
                        pred_demand = ema_demand[b][r]
                        scarcity += ((d[r] + pred_demand) / rem) ** 2

                # 3. Total cost
                cost = ALPHA * L[(b, s)] + BETA * carbon[s] + price_cost + gamma_s * scarcity

                if cost < best_cost:
                    best_cost, best_s = cost, s

        # apply allocation
        if best_s is not None:
            alloc[(b, d_idx)] = best_s
            for r in RESOURCES:
                remaining[(best_s, r)] -= d[r]

                # Dual update
                lam[(best_s, r)] += eta * (d[r] / capacity[(best_s, r)])
                lam[(best_s, r)] *= (1 - decay)
        else:
            alloc[(b, d_idx)] = None

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
    - After warmup: switches to predictive scarcity using EMA

    eta        = dual step size
    decay      = price stabilization
    gamma_s    = strength of scarcity penalty
    pred_alpha = EMA smoothing
    pred_window= required history before prediction kicks in
    """

    remaining = {(s, r): capacity[(s, r)] for s in sellers for r in RESOURCES}
    lam = {(s, r): 0.0 for s in sellers for r in RESOURCES}
    alloc = {}

    # history + EMA
    ema_demand = {b: {r: 0.0 for r in RESOURCES} for b, _, _ in stream}
    history = {b: [] for b, _, _ in stream}

    for (b, d_idx, d) in stream:

        # =========================
        # UPDATE HISTORY
        # =========================
        history[b].append(d)
        if len(history[b]) > pred_window:
            history[b].pop(0)

        # compute EMA
        for r in RESOURCES:
            past_vals = [h[r] for h in history[b]]
            ema = past_vals[0]
            for val in past_vals[1:]:
                ema = pred_alpha * val + (1 - pred_alpha) * ema
            ema_demand[b][r] = ema

        # =========================
        # CHECK IF WARMED UP
        # =========================
        use_prediction = len(history[b]) >= pred_window

        best_s, best_cost = None, float("inf")

        for s in sellers:
            if all(d[r] <= remaining[(s, r)] for r in RESOURCES):

                # 1. Lagrangian price
                price_cost = sum(
                    lam[(s, r)] * (d[r] / capacity[(s, r)])
                    for r in RESOURCES
                )

                # 2. Scarcity
                scarcity = 0.0
                for r in RESOURCES:
                    rem = remaining[(s, r)]
                    if rem > 0:
                        if use_prediction:
                            # predictive version
                            pred_demand = ema_demand[b][r]
                            scarcity += ((d[r] + pred_demand) / rem) ** 2
                        else:
                            # vanilla version
                            scarcity += (d[r] / rem) ** 2

                # 3. Total cost
                cost = (
                    ALPHA * L[(b, s)] +
                    BETA * carbon[s] +
                    price_cost +
                    gamma_s * scarcity
                )

                if cost < best_cost:
                    best_cost, best_s = cost, s

        # =========================
        # APPLY DECISION
        # =========================
        if best_s is not None:
            alloc[(b, d_idx)] = best_s

            for r in RESOURCES:
                remaining[(best_s, r)] -= d[r]

                # dual update (UNCHANGED)
                lam[(best_s, r)] += eta * (d[r] / capacity[(best_s, r)])
                lam[(best_s, r)] *= (1 - decay)

        else:
            alloc[(b, d_idx)] = None

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
    rejection_rate = rejected / total

    return avg_lat, avg_co2, p95_lat, rejection_rate


def compute_utilization_metrics(alloc, demands, capacity, sellers):
    used = {(s,r):0 for s in sellers for r in RESOURCES}

    # accumulate usage
    for (b,d_idx), s in alloc.items():
        if s is None:
            continue
        for r in RESOURCES:
            used[(s,r)] += demands[b][d_idx][r]

    utilizations = []

    for s in sellers:
        for r in RESOURCES:
            cap = capacity[(s,r)]
            if cap > 0:
                utilizations.append(used[(s,r)] / cap)

    if len(utilizations) == 0:
        return 0, 0

    return np.var(utilizations), np.max(utilizations)

# =====================================================
# MAIN
# =====================================================
# def main():

#     buyers, sellers, L = build_topology()
#     capacity, carbon = generate_capacities(sellers)
#     demands = generate_demands(buyers)

#     stream = flatten_demands(demands)

#     methods = {
#         "GlobalMILP": solve_milp(buyers, sellers, demands, capacity, carbon, L),
#         "SimpleGreedy": simple_greedy(buyers, sellers, demands, L, capacity),
#         "SmartGreedyV2": smart_greedy_v3(sellers, stream, L, capacity, carbon),
#         "RollingMILP+": rolling_milp(sellers, stream, capacity, carbon, L),
#         "BatchMILP+": batch_milp(sellers, stream, capacity, carbon, L),
#         "PricingV2": pricing_v3(sellers, stream, capacity, carbon, L),
#         "PrimalDual": primal_dual_online(sellers, stream, capacity, carbon, L),
#         "PrimalDualPred": primal_dual_online_pred(sellers, stream, capacity, carbon, L),
#         "RollingMILPPred": rolling_milp_pred(sellers, stream, capacity, carbon, L)
#     }

#     results = []
#     for name, alloc in methods.items():
#         avg_lat, avg_co2, p95_lat, rej = compute_stats(alloc, demands, carbon, L)
#         util_var, max_util = compute_utilization_metrics(alloc, demands, capacity, sellers)

#         results.append([
#             name,
#             avg_lat,
#             p95_lat,
#             avg_co2,
#             util_var,
#             max_util,
#             rej
#         ])

#     print("\n=== FINAL COMPARISON (FULL METRICS) ===")
#     print(tabulate(results,
#                 headers=[
#                     "Method",
#                     "Avg Lat",
#                     "P95 Lat",
#                     "Avg CO2",
#                     "Util Var",
#                     "Max Util",
#                     "Reject Rate"
#                 ],
#                 tablefmt="fancy_grid"))

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
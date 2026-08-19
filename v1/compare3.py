import networkx as nx
import numpy as np
import pulp as pl
from tabulate import tabulate

#np.random.seed(42)

RESOURCES = ["cpu","mem","gpu"]

ALPHA = 1.0
BETA  = 0.3
GAMMA = {"cpu":10, "mem":2, "gpu":20}


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
        carbon[s] = np.random.uniform(1,5)
    return capacity, carbon


def generate_demands(buyers, max_workloads=90):
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
# GLOBAL MILP
# =====================================================
def solve_milp(buyers, sellers, demands, capacity, carbon, L):
    model = pl.LpProblem("GlobalMILP", pl.LpMinimize)

    y = {(b,d_idx,s): pl.LpVariable(f"y_{b}_{d_idx}_{s}", cat="Binary")
         for b in buyers
         for d_idx in range(len(demands[b]))
         for s in sellers}

    for b in buyers:
        for d_idx in range(len(demands[b])):
            model += pl.lpSum(y[(b,d_idx,s)] for s in sellers) == 1

    for s in sellers:
        for r in RESOURCES:
            model += pl.lpSum(
                demands[b][d_idx][r] * y[(b,d_idx,s)]
                for b in buyers for d_idx in range(len(demands[b]))
            ) <= capacity[(s,r)]

    model += pl.lpSum(
        (ALPHA * L[(b,s)] + BETA * carbon[s]) * y[(b,d_idx,s)]
        for b in buyers for d_idx in range(len(demands[b])) for s in sellers
    )

    model.solve(pl.PULP_CBC_CMD(timeLimit=100))

    alloc = {}
    for b in buyers:
        for d_idx in range(len(demands[b])):
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
# SMART GREEDY V2 (IMPROVED)
# =====================================================
def smart_greedy(sellers, stream, L, capacity, carbon):
    remaining = {(s,r): capacity[(s,r)] for s in sellers for r in RESOURCES}
    alloc = {}

    max_latency = max(L.values())
    max_carbon = max(carbon.values())

    for (b,d_idx,d) in stream:
        best_s, best_cost = None, float("inf")

        for s in sellers:
            if all(d[r] <= remaining[(s,r)] for r in RESOURCES):

                lat = L[(b,s)] / max_latency
                co2 = carbon[s] / max_carbon

                scarcity = 0
                for r in RESOURCES:
                    util = 1 - remaining[(s,r)] / capacity[(s,r)]
                    scarcity += GAMMA[r] * (util**2) * (d[r] / capacity[(s,r)])

                cost = ALPHA*lat + BETA*co2 + scarcity

                if cost < best_cost:
                    best_cost, best_s = cost, s

        if best_s:
            alloc[(b,d_idx)] = best_s
            for r in RESOURCES:
                remaining[(best_s,r)] -= d[r]
        else:
            alloc[(b,d_idx)] = None

    return alloc


# =====================================================
# ROLLING MILP (IMPROVED)
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

        for j in range(len(batch)):
            model += pl.lpSum(y[(j,s)] for s in sellers) == 1

        for s in sellers:
            for r in RESOURCES:
                model += pl.lpSum(batch[j][2][r] * y[(j,s)]
                                  for j in range(len(batch))) <= remaining[(s,r)]

        model += pl.lpSum(
            (ALPHA*L[(batch[j][0],s)]
             + BETA*carbon[s]
             + sum(GAMMA[r]*(batch[j][2][r]/capacity[(s,r)]) for r in RESOURCES)
            ) * y[(j,s)]
            for j in range(len(batch)) for s in sellers
        )

        model.solve()

        b,d_idx,d = batch[0]
        for s in sellers:
            if y[(0,s)].value() > 0.5:
                alloc[(b,d_idx)] = s
                for r in RESOURCES:
                    remaining[(s,r)] -= d[r]

        i += 1

    return alloc


# =====================================================
# BATCH MILP (IMPROVED)
# =====================================================
def batch_milp(sellers, stream, capacity, carbon, L, BATCH_SIZE=10):
    remaining = {(s,r): capacity[(s,r)] for s in sellers for r in RESOURCES}
    alloc = {}

    for i in range(0, len(stream), BATCH_SIZE):
        batch = stream[i:i+BATCH_SIZE]

        model = pl.LpProblem("BatchMILP", pl.LpMinimize)

        y = {(j,s): pl.LpVariable(f"y_{j}_{s}", cat="Binary")
             for j in range(len(batch)) for s in sellers}

        for j in range(len(batch)):
            model += pl.lpSum(y[(j,s)] for s in sellers) == 1

        for s in sellers:
            for r in RESOURCES:
                model += pl.lpSum(batch[j][2][r] * y[(j,s)]
                                  for j in range(len(batch))) <= remaining[(s,r)]

        model += pl.lpSum(
            (ALPHA*L[(batch[j][0],s)]
             + BETA*carbon[s]
             + sum(GAMMA[r]*(batch[j][2][r]/capacity[(s,r)]) for r in RESOURCES)
            ) * y[(j,s)]
            for j in range(len(batch)) for s in sellers
        )

        model.solve()

        for j,(b,d_idx,d) in enumerate(batch):
            for s in sellers:
                if y[(j,s)].value() > 0.5:
                    alloc[(b,d_idx)] = s
                    for r in RESOURCES:
                        remaining[(s,r)] -= d[r]

    return alloc


# =====================================================
# PRICING V2 (IMPROVED)
# =====================================================
def pricing_method(sellers, stream, capacity, carbon, L):
    remaining = {(s,r): capacity[(s,r)] for s in sellers for r in RESOURCES}
    prices = {(s,r): 0.0 for s in sellers for r in RESOURCES}
    alloc = {}

    step_size = 0.05

    for (b,d_idx,d) in stream:
        best_s, best_cost = None, float("inf")

        for s in sellers:
            if all(d[r] <= remaining[(s,r)] for r in RESOURCES):

                price_cost = sum(prices[(s,r)] * d[r] for r in RESOURCES)
                cost = ALPHA*L[(b,s)] + BETA*carbon[s] + price_cost

                if cost < best_cost:
                    best_cost, best_s = cost, s

        if best_s:
            alloc[(b,d_idx)] = best_s

            for r in RESOURCES:
                remaining[(best_s,r)] -= d[r]

                util = 1 - remaining[(best_s,r)] / capacity[(best_s,r)]
                prices[(best_s,r)] += step_size * util

        else:
            alloc[(b,d_idx)] = None

    return alloc


# =====================================================
# STATS
# =====================================================
def compute_stats(alloc, demands, carbon, L):
    lat, co2 = [], []
    for (b,d_idx), s in alloc.items():
        if s:
            lat.append(L[(b,s)])
            co2.append(carbon[s])
    return np.mean(lat), np.mean(co2)


# =====================================================
# MAIN
# =====================================================
def main():

    buyers, sellers, L = build_topology()
    capacity, carbon = generate_capacities(sellers)
    demands = generate_demands(buyers)

    stream = flatten_demands(demands)

    methods = {
        "GlobalMILP": solve_milp(buyers, sellers, demands, capacity, carbon, L),
        "SimpleGreedy": simple_greedy(buyers, sellers, demands, L, capacity),
        "SmartGreedyV2": smart_greedy(sellers, stream, L, capacity, carbon),
        "RollingMILP+": rolling_milp(sellers, stream, capacity, carbon, L),
        "BatchMILP+": batch_milp(sellers, stream, capacity, carbon, L),
        "PricingV2": pricing_method(sellers, stream, capacity, carbon, L)
    }

    results = []
    for name, alloc in methods.items():
        lat, co2 = compute_stats(alloc, demands, carbon, L)
        results.append([name, lat, co2])

    print("\n=== FINAL COMPARISON (OPTIMIZED) ===")
    print(tabulate(results,
                   headers=["Method","Avg Latency","Avg Carbon"],
                   tablefmt="fancy_grid"))


if __name__ == "__main__":
    main()
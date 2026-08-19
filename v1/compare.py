import networkx as nx
import numpy as np
import pulp as pl
from tabulate import tabulate

#np.random.seed(42)

RESOURCES = ["cpu","mem","gpu"]

ALPHA = 1.0
BETA  = 0.3
GAMMA = {"cpu":10, "mem":2, "gpu":20}   # balancing weights


# =====================================================
# Topology + latency (expanded)
# =====================================================
def build_topology(num_buyers=5, num_sellers=20):
    buyers  = [f"B{i}" for i in range(num_buyers)]
    sellers = [f"S{i}" for i in range(num_sellers)]

    G = nx.Graph()
    
    # randomly connect buyers and sellers with latency between 1-20
    for b in buyers:
        connected_sellers = np.random.choice(sellers, size=5, replace=False)
        for s in connected_sellers:
            latency = np.random.randint(1,21)
            G.add_edge(b, s, weight=latency)
    
    # connect sellers in a chain for network propagation (optional)
    for i in range(num_sellers-1):
        G.add_edge(sellers[i], sellers[i+1], weight=np.random.randint(1,10))
    
    # compute latency matrix
    L = {}
    for b in buyers:
        lengths = nx.single_source_dijkstra_path_length(G, b, weight="weight")
        for s in sellers:
            # if not reachable, assign a high latency
            L[(b,s)] = lengths.get(s, 50)
    
    return buyers, sellers, L


def print_latency_matrix(buyers, sellers, L):
    table = []
    for b in buyers:
        row = [b] + [L[(b,s)] for s in sellers]
        table.append(row)

    print("\n=== Latency Matrix ===")
    print(tabulate(table, headers=[""]+sellers, tablefmt="fancy_grid"))



# =====================================================
# Data generation
# =====================================================
def generate_capacities(sellers):
    capacity = {}
    carbon = {}

    for s in sellers:
        capacity[(s,"cpu")] = np.random.randint(32,129)
        capacity[(s,"mem")] = np.random.randint(128,513)
        capacity[(s,"gpu")] = np.random.randint(1,9)  # GPU >=1
        carbon[s] = np.random.uniform(1,5)

    return capacity, carbon


def generate_demands(buyers, max_workloads=60):
    demands = {b: [] for b in buyers}

    for _ in range(max_workloads):
        b = np.random.choice(buyers)
        demand = {
            "cpu": np.random.randint(2,31),
            "mem": np.random.randint(4,33),
            "gpu": np.random.randint(0,3)  # GPU can be 0
        }
        demands[b].append(demand)

    return demands

# =====================================================
# Pretty printing
# =====================================================
def print_capacities(capacity, carbon, sellers):
    table = []
    for s in sellers:
        table.append([
            s,
            capacity[(s,"cpu")],
            capacity[(s,"mem")],
            capacity[(s,"gpu")],
            round(carbon[s],2)
        ])

    print("\n=== Seller Capacities ===")
    print(tabulate(table,
                   headers=["Seller","CPU","MEM","GPU","Carbon"],
                   tablefmt="fancy_grid"))


def print_demands(demands):
    table = []
    for b, ds in demands.items():
        for i,d in enumerate(ds):
            table.append([f"{b}-D{i}", d["cpu"], d["mem"], d["gpu"]])

    print("\n=== Buyer Demands ===")
    print(tabulate(table,
                   headers=["Demand","CPU","MEM","GPU"],
                   tablefmt="fancy_grid"))


def print_util(title, used, capacity, sellers):
    table = []
    for s in sellers:
        row = [s]
        for r in RESOURCES:
            u = used[(s,r)]
            c = capacity[(s,r)]
            row.append(f"{u}/{c} ({u/c:.2f})")
        table.append(row)

    print(f"\n=== {title} ===")
    print(tabulate(table,
                   headers=["Seller","CPU","MEM","GPU"],
                   tablefmt="fancy_grid"))


# =====================================================
# MILP (NO SPLIT + BALANCING)
# =====================================================
def solve_milp(buyers, sellers, demands, capacity, carbon, L):

    model = pl.LpProblem("MarketplaceMILP", pl.LpMinimize)

    y = {
        b: {
            d_idx: {
                s: pl.LpVariable(f"y_{b}_{d_idx}_{s}", cat="Binary")
                for s in sellers
            }
            for d_idx in range(len(demands[b]))
        }
        for b in buyers
    }

    # each demand exactly one seller
    for b in buyers:
        for d_idx in range(len(demands[b])):
            model += pl.lpSum(y[b][d_idx][s] for s in sellers) == 1

    # capacity constraints
    for s in sellers:
        for r in RESOURCES:
            model += pl.lpSum(
                demands[b][d_idx][r] * y[b][d_idx][s]
                for b in buyers
                for d_idx in range(len(demands[b]))
            ) <= capacity[(s,r)]

    latency_cost = pl.lpSum(
        L[(b,s)] * y[b][d_idx][s]
        for b in buyers for d_idx in range(len(demands[b])) for s in sellers
    )

    carbon_cost = pl.lpSum(
        carbon[s] * y[b][d_idx][s]
        for b in buyers for d_idx in range(len(demands[b])) for s in sellers
    )

    util_cost = pl.lpSum(
        GAMMA[r] *
        (
            pl.lpSum(demands[b][d_idx][r] * y[b][d_idx][s]
                     for b in buyers for d_idx in range(len(demands[b])))
            / capacity[(s,r)]
        )
        for s in sellers for r in RESOURCES
    )

    model += ALPHA*latency_cost + BETA*carbon_cost + util_cost

    model.solve()

    alloc = {}
    for b in buyers:
        for d_idx in range(len(demands[b])):
            for s in sellers:
                if y[b][d_idx][s].value() > 0.5:
                    alloc[(b,d_idx)] = s

    return alloc


# =====================================================
# Greedy assignment with capacity check
# =====================================================
def greedy_assignment(buyers, sellers, demands, L, capacity):
    # Track remaining capacities
    remaining = {(s,r): capacity[(s,r)] for s in sellers for r in RESOURCES}
    alloc = {}

    for b in buyers:
        for d_idx, demand in enumerate(demands[b]):
            # Sort sellers by latency for this buyer
            sorted_sellers = sorted(sellers, key=lambda s: L[(b,s)])
            
            assigned = False
            for s in sorted_sellers:
                # Check if seller can accommodate this demand
                fits = all(demand[r] <= remaining[(s,r)] for r in RESOURCES)
                if fits:
                    alloc[(b,d_idx)] = s
                    # Deduct capacities
                    for r in RESOURCES:
                        remaining[(s,r)] -= demand[r]
                    assigned = True
                    break

            if not assigned:
                # If no seller can fit, assign None (or handle specially)
                alloc[(b,d_idx)] = None
                print(f"Warning: Demand {b}-{d_idx} could not be assigned due to capacity limits.")

    return alloc

# =====================================================
# Stats
# =====================================================
def compute_stats(alloc, demands, carbon, L):
    lat = []
    co2 = []
    for (b,d_idx), s in alloc.items():
        lat.append(L[(b,s)])
        co2.append(carbon[s])
    return np.mean(lat), np.mean(co2)


def compute_util(alloc, demands, sellers):
    used = {(s,r):0 for s in sellers for r in RESOURCES}
    for (b,d_idx), s in alloc.items():
        for r in RESOURCES:
            used[(s,r)] += demands[b][d_idx][r]
    return used


# =====================================================
# MAIN
# =====================================================
def main():

    buyers, sellers, L = build_topology()

    print_latency_matrix(buyers, sellers, L)

    capacity, carbon = generate_capacities(sellers)
    demands = generate_demands(buyers)

    print_capacities(capacity, carbon, sellers)
    print_demands(demands)

    milp_alloc   = solve_milp(buyers, sellers, demands, capacity, carbon, L)
    greedy_alloc = greedy_assignment(buyers, sellers, demands, L, capacity)

    milp_lat, milp_co2 = compute_stats(milp_alloc, demands, carbon, L)
    greedy_lat, greedy_co2 = compute_stats(greedy_alloc, demands, carbon, L)

    print("\n=== Performance Comparison ===")
    print(tabulate([
        ["MILP",   milp_lat,   milp_co2],
        ["Greedy", greedy_lat, greedy_co2]
    ],
    headers=["Method","Avg Latency","Avg Carbon"],
    tablefmt="fancy_grid"))

    print_util("MILP Utilization",
               compute_util(milp_alloc, demands, sellers),
               capacity, sellers)

    print_util("Greedy Utilization",
               compute_util(greedy_alloc, demands, sellers),
               capacity, sellers)


if __name__ == "__main__":
    main()
"""
Simulated seller world -- direct port of build_topology()/generate_capacities()
from the validated v2 simulation script, restructured so it can serve an
open-ended, real-time stream of buyers (not a fixed pre-generated buyer
list) while staying fully reproducible.

*** THIS IS THE PART THE USER SAID WILL BE REPLACED BY REAL SELLER
    DISCOVERY LATER. *** Everything else in this codebase only talks to
    this module through the SimulatedWorld interface below (sellers,
    capacity, carbon, latency_for_buyer), so swapping it for a real
    discovery/topology service later should not require touching the
    algorithm or worker code -- just this file (or a drop-in replacement
    behind the same interface).

Design note on fairness: all 4 algorithm workers run as separate
processes with their own private, independently-mutated `remaining`
capacity dict (this is intentional -- see the accompanying design
discussion). For the four-way comparison to mean anything, every worker
must still see an *identical* initial world: same seller capacities,
same carbon values, same buyer<->seller latencies. That's achieved here
by deriving every random draw from WORLD_SEED (global) and, for
per-buyer latency, from a deterministic hash of (WORLD_SEED, buyer_id) --
so any process, in any order, computes the same latency dict for a given
buyer_id without needing to share state at runtime.
"""
import hashlib
import threading

import networkx as nx
import numpy as np

from seller_workload_opt.v3.common.config import WORLD_SEED, NUM_SELLERS, BUYER_DEGREE, RESOURCES
from seller_workload_opt.v3.common.logging_setup import get_logger

log = get_logger("world")


def _buyer_seed(world_seed: int, buyer_id: str) -> int:
    digest = hashlib.sha256(f"{world_seed}:{buyer_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % (2 ** 32)


class SimulatedWorld:
    def __init__(self, num_sellers=NUM_SELLERS, seed=WORLD_SEED, buyer_degree=BUYER_DEGREE):
        self.num_sellers = num_sellers
        self.seed = seed
        self.buyer_degree = buyer_degree
        self.sellers = [f"S{i}" for i in range(num_sellers)]

        self._build_seller_graph()
        self._generate_capacity_and_carbon()

        self._latency_cache = {}
        self._latency_lock = threading.Lock()

        log.info(
            f"SimulatedWorld ready: seed={seed} num_sellers={num_sellers} "
            f"total_cpu_capacity={sum(self.capacity[(s, 'cpu')] for s in self.sellers)} "
            f"avg_carbon={np.mean(list(self.carbon.values())):.2f}"
        )

    # -- seller<->seller backbone graph (static, deterministic from seed) --
    def _build_seller_graph(self):
        rng = np.random.RandomState(self.seed)
        G = nx.Graph()
        G.add_nodes_from(self.sellers)
        for i in range(self.num_sellers - 1):
            G.add_edge(self.sellers[i], self.sellers[i + 1], weight=int(rng.randint(1, 10)))
        # a handful of extra edges so the backbone isn't just a single chain
        extra_edges = max(1, self.num_sellers // 3)
        for _ in range(extra_edges):
            a, b = rng.choice(self.sellers, size=2, replace=False)
            G.add_edge(a, b, weight=int(rng.randint(1, 10)))
        self.graph = G
        log.info(
            f"Seller backbone graph built: {self.num_sellers} sellers, "
            f"{G.number_of_edges()} edges"
        )

    # -- static per-seller capacity + carbon (deterministic from seed) -----
    def _generate_capacity_and_carbon(self):
        rng = np.random.RandomState(self.seed + 1)
        self.capacity = {}
        self.carbon = {}
        for s in self.sellers:
            self.capacity[(s, "cpu")] = int(rng.randint(32, 129))
            self.capacity[(s, "mem")] = int(rng.randint(128, 513))
            self.capacity[(s, "gpu")] = int(rng.randint(1, 9))
            self.carbon[s] = float(rng.uniform(1, 10))

    def initial_remaining(self):
        """A fresh copy of full seller capacity -- callers mutate their own copy."""
        return dict(self.capacity)

    def max_latency_estimate(self):
        # a generous static upper bound for normalizing costs (edge weights are 1-20)
        return 20 + 10  # buyer edge (<=20) + a couple of backbone hops

    def max_carbon(self):
        return max(self.carbon.values())

    # -- per-buyer latency, deterministic + cached ---------------------------
    def latency_for_buyer(self, buyer_id: str) -> dict:
        """
        Returns {seller: latency} for this buyer. Deterministic given
        (WORLD_SEED, buyer_id): any process computes the identical dict,
        which is what keeps the 4 independent workers comparable without
        sharing runtime state.
        """
        with self._latency_lock:
            cached = self._latency_cache.get(buyer_id)
            if cached is not None:
                return cached

        seed = _buyer_seed(self.seed, buyer_id)
        rng = np.random.RandomState(seed)
        deg = min(self.buyer_degree, self.num_sellers)
        chosen = rng.choice(self.sellers, size=deg, replace=False)

        G = self.graph.copy()
        for s in chosen:
            G.add_edge(buyer_id, s, weight=int(rng.randint(1, 21)))

        lengths = nx.single_source_dijkstra_path_length(G, buyer_id, weight="weight")
        result = {s: lengths.get(s, 50) for s in self.sellers}

        with self._latency_lock:
            self._latency_cache[buyer_id] = result
        return result


_world_instance = None
_world_lock = threading.Lock()


def get_world() -> SimulatedWorld:
    """Process-local singleton -- one SimulatedWorld per worker process."""
    global _world_instance
    with _world_lock:
        if _world_instance is None:
            _world_instance = SimulatedWorld()
        return _world_instance

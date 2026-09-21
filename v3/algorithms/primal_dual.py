"""
PrimalDual -- direct port of primal_dual_online() from the validated v2
simulation. Pure online: one demand item in, one decision out, no
buffering/lookahead. Dual prices (`lam`) and `remaining` capacity are
private state on this processor instance.
"""
import time

from seller_workload_opt.v3.common.config import RESOURCES, PD_ETA, PD_DECAY, PD_GAMMA_S, ALGORITHMS
from seller_workload_opt.v3.algorithms.common import cost_of


class PrimalDualProcessor:
    name = "primal_dual"

    def __init__(self, world, logger, eta=PD_ETA, decay=PD_DECAY, gamma_s=PD_GAMMA_S):
        self.world = world
        self.log = logger
        self.eta = eta
        self.decay = decay
        self.gamma_s = gamma_s

        self.remaining = world.initial_remaining()
        self.lam = {(s, r): 0.0 for s in world.sellers for r in RESOURCES}
        self.n_processed = 0
        self.n_rejected = 0

        self.log.info(
            f"PrimalDualProcessor initialized: eta={eta} decay={decay} "
            f"gamma_s={gamma_s} sellers={len(world.sellers)}"
        )

    def step(self, msg_id, item):
        """Immediate decision -- always returns exactly one finalized result."""
        t0 = time.time()
        d = item.demand()
        best_s, best_cost = None, float("inf")

        for s in self.world.sellers:
            if all(d[r] <= self.remaining[(s, r)] for r in RESOURCES):
                price_cost = sum(
                    self.lam[(s, r)] * (d[r] / self.world.capacity[(s, r)]) for r in RESOURCES
                )
                scarcity = 0.0
                for r in RESOURCES:
                    rem = self.remaining[(s, r)]
                    if rem > 0:
                        scarcity += (d[r] / rem) ** 2
                _, _, base_cost = cost_of(item.buyer_id, s, self.world)
                cost = base_cost + price_cost + self.gamma_s * scarcity
                if cost < best_cost:
                    best_cost, best_s = cost, s

        solve_ms = (time.time() - t0) * 1000
        self.n_processed += 1

        if best_s is not None:
            for r in RESOURCES:
                self.remaining[(best_s, r)] -= d[r]
                self.lam[(best_s, r)] += self.eta * (d[r] / self.world.capacity[(best_s, r)])
                self.lam[(best_s, r)] *= (1 - self.decay)
            lat_cost, carbon_cost, total_cost = cost_of(item.buyer_id, best_s, self.world)
            decision = {
                "seller": best_s,
                "rejected": False,
                "lat_cost": lat_cost,
                "carbon_cost": carbon_cost,
                "total_cost": total_cost,
                "solve_ms": solve_ms,
                "window_id": f"item-{item.demand_id}",
                "status": "Online",
                "window_members": [item.demand_id],
            }
            self.log.info(
                f"demand_id={item.demand_id} buyer={item.buyer_id} -> seller={best_s} "
                f"demand={d} total_cost={total_cost:.3f} avg_lambda="
                f"{sum(self.lam.values()) / len(self.lam):.4f} solve_ms={solve_ms:.2f}"
            )
        else:
            self.n_rejected += 1
            decision = {
                "seller": None,
                "rejected": True,
                "lat_cost": 0.0,
                "carbon_cost": 0.0,
                "total_cost": None,  # not float("inf") -- Infinity isn't valid JSON
                "solve_ms": solve_ms,
                "window_id": f"item-{item.demand_id}",
                "status": "Infeasible",
                "window_members": [item.demand_id],
            }
            self.log.warning(
                f"REJECTED demand_id={item.demand_id} buyer={item.buyer_id} demand={d} "
                f"-- no seller has sufficient remaining capacity "
                f"(processed={self.n_processed} rejected={self.n_rejected})"
            )

        if self.n_processed % 20 == 0:
            self.log.info(
                f"progress: processed={self.n_processed} rejected={self.n_rejected} "
                f"reject_rate={self.n_rejected / self.n_processed:.3f}"
            )

        return [(msg_id, item, decision)]

    def tick(self):
        return []  # nothing time-based to flush; every item decides immediately

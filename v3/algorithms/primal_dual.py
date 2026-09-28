"""
PrimalDual -- direct port of primal_dual_online() from the validated v2
simulation. Pure online: one demand item in, one decision out, no
buffering/lookahead. Dual prices (`lam`) and `remaining` capacity are
private state on this processor instance.

Capacity now comes from the live SellerWorld (real sellers) instead of a
fixed simulated snapshot -- resync_remaining() is called at the top of
every step() to reconcile against the latest known telemetry (see
algorithms/common.py and common/world.py for the full policy). Sellers
are also discovered dynamically now (they may not exist yet when this
processor starts, or may go offline later), so `lam` is built lazily
per-seller instead of being seeded once from a fixed seller list at init.
"""
import time

from seller_workload_opt.v3.common.config import RESOURCES, PD_ETA, PD_DECAY, PD_GAMMA_S
from seller_workload_opt.v3.algorithms.common import cost_of, safe_ratio, resync_remaining


class PrimalDualProcessor:
    name = "primal_dual"

    def __init__(self, world, logger, eta=PD_ETA, decay=PD_DECAY, gamma_s=PD_GAMMA_S):
        self.world = world
        self.log = logger
        self.eta = eta
        self.decay = decay
        self.gamma_s = gamma_s

        self.remaining = {}
        self.last_synced = {}
        self.lam = {}
        self.n_processed = 0
        self.n_rejected = 0

        self.log.info(f"PrimalDualProcessor initialized: eta={eta} decay={decay} gamma_s={gamma_s}")

    def _lam(self, s, r):
        return self.lam.get((s, r), 0.0)

    def step(self, msg_id, item):
        """Immediate decision -- always returns exactly one finalized result."""
        t0 = time.time()
        resync_remaining(self.remaining, self.last_synced, self.world)

        d = item.demand()
        sellers = self.world.sellers
        capacity = self.world.capacity
        best_s, best_cost = None, float("inf")

        for s in sellers:
            if all(d[r] <= self.remaining.get((s, r), 0.0) for r in RESOURCES):
                price_cost = sum(
                    self._lam(s, r) * safe_ratio(d[r], capacity.get((s, r), 0.0)) for r in RESOURCES
                )
                scarcity = 0.0
                for r in RESOURCES:
                    rem = self.remaining.get((s, r), 0.0)
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
                new_lam = self._lam(best_s, r) + self.eta * safe_ratio(d[r], capacity.get((best_s, r), 0.0))
                self.lam[(best_s, r)] = new_lam * (1 - self.decay)
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
            avg_lambda = (sum(self.lam.values()) / len(self.lam)) if self.lam else 0.0
            self.log.info(
                f"demand_id={item.demand_id} buyer={item.buyer_id} -> seller={best_s} "
                f"demand={d} total_cost={total_cost:.3f} avg_lambda={avg_lambda:.4f} "
                f"solve_ms={solve_ms:.2f} known_sellers={len(sellers)}"
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
                f"(known_sellers={len(sellers)} processed={self.n_processed} rejected={self.n_rejected})"
            )

        if self.n_processed % 20 == 0:
            self.log.info(
                f"progress: processed={self.n_processed} rejected={self.n_rejected} "
                f"reject_rate={self.n_rejected / self.n_processed:.3f} known_sellers={len(sellers)}"
            )

        return [(msg_id, item, decision)]

    def tick(self):
        return []  # nothing time-based to flush; every item decides immediately

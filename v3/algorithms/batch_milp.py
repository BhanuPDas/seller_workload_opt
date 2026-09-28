"""
BatchMILP+ -- streaming adaptation of batch_milp() from the validated v2
simulation. Non-overlapping batches: buffer until BATCH_SIZE items have
arrived (or a max-wait timeout fires), solve the whole buffered batch as
one MILP, commit every item in it, and start the next batch empty.
Simpler than RollingMILP+ (no carryover buffer between solves).
"""
import time

import pulp as pl

from seller_workload_opt.v3.common.config import (
    RESOURCES, ALPHA, BETA, REJECTION_PENALTY,
    BATCH_SIZE, BATCH_MAX_WAIT_SECONDS, BATCH_SOLVE_TIME_LIMIT,
)
from seller_workload_opt.v3.algorithms.common import binval, solved_ok, cost_of, rejected_decision, resync_remaining


class BatchMilpProcessor:
    name = "batch_milp"

    def __init__(self, world, logger, batch_size=BATCH_SIZE,
                 max_wait=BATCH_MAX_WAIT_SECONDS, time_limit=BATCH_SOLVE_TIME_LIMIT):
        self.world = world
        self.log = logger
        self.batch_size = batch_size
        self.max_wait = max_wait
        self.time_limit = time_limit

        self.remaining = {}
        self.last_synced = {}
        self.buffer = []
        self.first_buffered_at = None
        self.batch_idx = 0
        self.n_committed = 0
        self.n_rejected = 0

        self.log.info(
            f"BatchMilpProcessor initialized: batch_size={batch_size} "
            f"max_wait={max_wait}s time_limit={time_limit}s"
        )

    def step(self, msg_id, item):
        if not self.buffer:
            self.first_buffered_at = time.time()
        self.buffer.append((msg_id, item))
        self.log.info(
            f"buffered demand_id={item.demand_id} buyer={item.buyer_id} "
            f"batch_fill={len(self.buffer)}/{self.batch_size}"
        )
        if len(self.buffer) >= self.batch_size:
            return self._solve(reason="batch_full")
        return []

    def tick(self):
        if self.buffer and self.first_buffered_at is not None:
            waited = time.time() - self.first_buffered_at
            if waited >= self.max_wait:
                self.log.info(
                    f"max_wait={self.max_wait}s elapsed with a partial batch "
                    f"({len(self.buffer)}/{self.batch_size}) -- forcing a solve"
                )
                return self._solve(reason="max_wait_timeout")
        return []

    def _solve(self, reason):
        resync_remaining(self.remaining, self.last_synced, self.world)
        sellers = self.world.sellers
        carbon = self.world.carbon

        batch = self.buffer
        batch_id = f"batch-{self.batch_idx}"
        member_ids = [it.demand_id for _, it in batch]

        if not sellers:
            self.log.warning(
                f"batch_id={batch_id}: 0 sellers known -- every item in this batch will "
                f"be rejected. Check that sellers are publishing to the seller stream."
            )

        model = pl.LpProblem("BatchMILP", pl.LpMinimize)
        y = {(j, s): pl.LpVariable(f"y_{j}_{s}", cat="Binary")
             for j in range(len(batch)) for s in sellers}
        z = {j: pl.LpVariable(f"z_{j}", cat="Binary") for j in range(len(batch))}

        for j in range(len(batch)):
            model += pl.lpSum(y[(j, s)] for s in sellers) == z[j]

        for s in sellers:
            for r in RESOURCES:
                model += pl.lpSum(
                    batch[j][1].demand()[r] * y[(j, s)] for j in range(len(batch))
                ) <= self.remaining.get((s, r), 0.0)

        model += (
            pl.lpSum(
                (ALPHA * self.world.latency_for_buyer(batch[j][1].buyer_id)[s] + BETA * carbon[s])
                * y[(j, s)]
                for j in range(len(batch)) for s in sellers
            )
            + pl.lpSum(REJECTION_PENALTY * (1 - z[j]) for j in range(len(batch)))
        )

        self.log.info(
            f"SOLVING batch_id={batch_id} reason={reason} size={len(batch)} members={member_ids}"
        )
        t0 = time.time()
        model.solve(pl.PULP_CBC_CMD(timeLimit=self.time_limit, msg=0))
        solve_ms = (time.time() - t0) * 1000
        ok = solved_ok(model)
        status = pl.LpStatus[model.status]
        if not ok:
            self.log.warning(
                f"batch_id={batch_id} solver status={status} (not Optimal) within "
                f"{self.time_limit}s -- unresolved items in this batch are treated as rejected"
            )
        self.log.info(f"SOLVED batch_id={batch_id} status={status} solve_ms={solve_ms:.1f}")

        results = []
        batch_rejected = 0
        for j, (j_msg_id, it) in enumerate(batch):
            d = it.demand()
            if not ok or binval(z[j]) < 0.5:
                decision = rejected_decision(batch_id, member_ids, status, solve_ms)
                batch_rejected += 1
                self.n_rejected += 1
                self.log.warning(
                    f"REJECTED demand_id={it.demand_id} buyer={it.buyer_id} demand={d} in {batch_id}"
                )
            else:
                chosen = None
                for s in sellers:
                    if binval(y[(j, s)]) > 0.5:
                        chosen = s
                        break
                for r in RESOURCES:
                    self.remaining[(chosen, r)] -= d[r]
                lat_cost, carbon_cost, total_cost = cost_of(it.buyer_id, chosen, self.world)
                decision = {
                    "seller": chosen, "rejected": False,
                    "lat_cost": lat_cost, "carbon_cost": carbon_cost, "total_cost": total_cost,
                    "solve_ms": solve_ms, "window_id": batch_id, "status": status,
                    "window_members": member_ids,
                }
                self.n_committed += 1
                self.log.info(
                    f"COMMITTED demand_id={it.demand_id} buyer={it.buyer_id} -> seller={chosen} "
                    f"demand={d} total_cost={total_cost:.3f} batch_id={batch_id}"
                )
            results.append((j_msg_id, it, decision))

        self.buffer = []
        self.first_buffered_at = None
        self.batch_idx += 1
        self.log.info(
            f"batch_id={batch_id} DONE: committed={len(batch) - batch_rejected} "
            f"rejected={batch_rejected} totals(committed={self.n_committed}, rejected={self.n_rejected})"
        )
        return results

"""
RollingMILP+ -- streaming adaptation of rolling_milp() from the validated
v2 simulation. Buffers arrivals into a sliding window of up to K items,
solves a joint MILP over the window once S NEW items have arrived (or a
max-wait timeout fires so a quiet period doesn't stall the window
forever), commits the first `commit_n` items' decisions, and keeps the
remainder buffered for the next window.
"""
import time

import pulp as pl

from seller_workload_opt.v3.common.config import (
    RESOURCES, ALPHA, BETA, REJECTION_PENALTY,
    ROLLING_K, ROLLING_S, ROLLING_MAX_WAIT_SECONDS, ROLLING_SOLVE_TIME_LIMIT,
)
from seller_workload_opt.v3.algorithms.common import binval, solved_ok, cost_of, rejected_decision


class RollingMilpProcessor:
    name = "rolling_milp"

    def __init__(self, world, logger, K=ROLLING_K, S=ROLLING_S,
                 max_wait=ROLLING_MAX_WAIT_SECONDS, time_limit=ROLLING_SOLVE_TIME_LIMIT):
        assert 1 <= S <= K, "S must satisfy 1 <= S <= K"
        self.world = world
        self.log = logger
        self.K = K
        self.S = S
        self.max_wait = max_wait
        self.time_limit = time_limit

        self.remaining = world.initial_remaining()
        self.buffer = []          # list of (msg_id, DemandItem), oldest first
        self.since_last_solve = 0
        self.last_solve_time = time.time()
        self.window_idx = 0
        self.n_committed = 0
        self.n_rejected = 0

        self.log.info(
            f"RollingMilpProcessor initialized: K={K} S={S} "
            f"max_wait={max_wait}s time_limit={time_limit}s sellers={len(world.sellers)}"
        )

    def step(self, msg_id, item):
        self.buffer.append((msg_id, item))
        self.since_last_solve += 1
        self.log.info(
            f"buffered demand_id={item.demand_id} buyer={item.buyer_id} "
            f"buffer_size={len(self.buffer)} since_last_solve={self.since_last_solve}/{self.S}"
        )
        if self.since_last_solve >= self.S:
            return self._solve(reason="count_threshold")
        return []

    def tick(self):
        if self.buffer and (time.time() - self.last_solve_time) >= self.max_wait:
            self.log.info(
                f"max_wait={self.max_wait}s elapsed with {len(self.buffer)} buffered "
                f"({self.since_last_solve} new since last solve) -- forcing a partial solve"
            )
            return self._solve(reason="max_wait_timeout")
        return []

    def _solve(self, reason):
        window = self.buffer[: self.K]
        commit_n = min(self.S, len(window))
        window_id = f"rolling-w{self.window_idx}"
        member_ids = [it.demand_id for _, it in window]

        model = pl.LpProblem("RollingMILP", pl.LpMinimize)
        y = {(j, s): pl.LpVariable(f"y_{j}_{s}", cat="Binary")
             for j in range(len(window)) for s in self.world.sellers}
        z = {j: pl.LpVariable(f"z_{j}", cat="Binary") for j in range(len(window))}

        for j in range(len(window)):
            model += pl.lpSum(y[(j, s)] for s in self.world.sellers) == z[j]

        for s in self.world.sellers:
            for r in RESOURCES:
                model += pl.lpSum(
                    window[j][1].demand()[r] * y[(j, s)] for j in range(len(window))
                ) <= self.remaining[(s, r)]

        model += (
            pl.lpSum(
                (ALPHA * self.world.latency_for_buyer(window[j][1].buyer_id)[s] + BETA * self.world.carbon[s])
                * y[(j, s)]
                for j in range(len(window)) for s in self.world.sellers
            )
            + pl.lpSum(REJECTION_PENALTY * (1 - z[j]) for j in range(len(window)))
        )

        self.log.info(
            f"SOLVING window_id={window_id} reason={reason} size={len(window)} "
            f"committing={commit_n} members={member_ids}"
        )
        t0 = time.time()
        model.solve(pl.PULP_CBC_CMD(timeLimit=self.time_limit, msg=0))
        solve_ms = (time.time() - t0) * 1000
        ok = solved_ok(model)
        status = pl.LpStatus[model.status]
        if not ok:
            self.log.warning(
                f"window_id={window_id} solver status={status} (not Optimal) within "
                f"{self.time_limit}s -- unresolved items in this window are treated as rejected"
            )
        self.log.info(f"SOLVED window_id={window_id} status={status} solve_ms={solve_ms:.1f}")

        results = []
        window_rejected = 0
        for j in range(commit_n):
            j_msg_id, it = window[j]
            d = it.demand()
            if not ok or binval(z[j]) < 0.5:
                decision = rejected_decision(window_id, member_ids, status, solve_ms)
                window_rejected += 1
                self.n_rejected += 1
                self.log.warning(
                    f"REJECTED demand_id={it.demand_id} buyer={it.buyer_id} demand={d} "
                    f"in {window_id} (window infeasible/unsolved for this item)"
                )
            else:
                chosen = None
                for s in self.world.sellers:
                    if binval(y[(j, s)]) > 0.5:
                        chosen = s
                        break
                for r in RESOURCES:
                    self.remaining[(chosen, r)] -= d[r]
                lat_cost, carbon_cost, total_cost = cost_of(it.buyer_id, chosen, self.world)
                decision = {
                    "seller": chosen, "rejected": False,
                    "lat_cost": lat_cost, "carbon_cost": carbon_cost, "total_cost": total_cost,
                    "solve_ms": solve_ms, "window_id": window_id, "status": status,
                    "window_members": member_ids,
                }
                self.n_committed += 1
                self.log.info(
                    f"COMMITTED demand_id={it.demand_id} buyer={it.buyer_id} -> seller={chosen} "
                    f"demand={d} total_cost={total_cost:.3f} window_id={window_id}"
                )
            results.append((j_msg_id, it, decision))

        self.buffer = self.buffer[commit_n:]
        self.since_last_solve = 0
        self.last_solve_time = time.time()
        self.window_idx += 1
        self.log.info(
            f"window_id={window_id} DONE: committed={commit_n - window_rejected} "
            f"rejected={window_rejected} remaining_buffer={len(self.buffer)} "
            f"totals(committed={self.n_committed}, rejected={self.n_rejected})"
        )
        return results

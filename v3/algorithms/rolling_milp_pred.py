"""
RollingMILPPred -- streaming adaptation of rolling_milp_pred() (the v2,
soft-penalty version) from the validated simulation. Same sliding-window
mechanics as RollingMILP+, plus a per-buyer EMA demand forecast that adds
a *soft* cost penalty (never a hard capacity reservation -- that was the
v1 bug this rewrite fixed) steering allocation away from sellers a buyer
is likely to need again soon.

Buyers are discovered on the fly (unlike the offline simulation, which
knew the buyer list upfront) -- ema_demand/history_count entries are
created lazily on first sight of a buyer_id.
"""
import time

import pulp as pl

from seller_workload_opt.v3.common.config import (
    RESOURCES, ALPHA, BETA, REJECTION_PENALTY,
    PRED_K, PRED_S, PRED_MAX_WAIT_SECONDS, PRED_SOLVE_TIME_LIMIT,
    PRED_W, PRED_F, PRED_W_MIN, PRED_WEIGHT, PRED_EMA_ALPHA,
)
from seller_workload_opt.v3.algorithms.common import binval, solved_ok, cost_of, rejected_decision


class RollingMilpPredProcessor:
    name = "rolling_milp_pred"

    def __init__(self, world, logger, K=PRED_K, S=PRED_S, max_wait=PRED_MAX_WAIT_SECONDS,
                 time_limit=PRED_SOLVE_TIME_LIMIT, W=PRED_W, F=PRED_F, W_min=PRED_W_MIN,
                 pred_weight=PRED_WEIGHT, ema_alpha=PRED_EMA_ALPHA):
        assert 1 <= S <= K, "S must satisfy 1 <= S <= K"
        self.world = world
        self.log = logger
        self.K = K
        self.S = S
        self.max_wait = max_wait
        self.time_limit = time_limit
        self.W = W
        self.F = F
        self.W_min = W_min
        self.pred_weight = pred_weight
        self.ema_alpha = ema_alpha

        self.remaining = world.initial_remaining()
        self.buffer = []
        self.since_last_solve = 0
        self.last_solve_time = time.time()
        self.window_idx = 0
        self.n_committed = 0
        self.n_rejected = 0

        self.ema_demand = {}      # buyer_id -> {r: ema}
        self.history_count = {}   # buyer_id -> int

        self.log.info(
            f"RollingMilpPredProcessor initialized: K={K} S={S} max_wait={max_wait}s "
            f"W={W} F={F} W_min={W_min} pred_weight={pred_weight} ema_alpha={ema_alpha}"
        )

    def _ensure_buyer(self, buyer_id):
        if buyer_id not in self.ema_demand:
            self.ema_demand[buyer_id] = {r: 0.0 for r in RESOURCES}
            self.history_count[buyer_id] = 0

    def step(self, msg_id, item):
        self._ensure_buyer(item.buyer_id)
        self.buffer.append((msg_id, item))
        self.since_last_solve += 1
        self.log.info(
            f"buffered demand_id={item.demand_id} buyer={item.buyer_id} "
            f"buffer_size={len(self.buffer)} since_last_solve={self.since_last_solve}/{self.S} "
            f"buyer_history_count={self.history_count[item.buyer_id]}"
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

    def _forecast_for(self, buyer_id):
        hc = self.history_count.get(buyer_id, 0)
        conf = min(hc / self.W, 1.0) if hc >= self.W_min else 0.0
        return {r: self.ema_demand[buyer_id][r] * self.F * conf for r in RESOURCES}

    def _solve(self, reason):
        window = self.buffer[: self.K]
        commit_n = min(self.S, len(window))
        window_id = f"pred-w{self.window_idx}"
        member_ids = [it.demand_id for _, it in window]

        buyers_in_window = {it.buyer_id for _, it in window}
        forecast = {b: self._forecast_for(b) for b in buyers_in_window}
        max_forecast_cpu = max((forecast[b]["cpu"] for b in buyers_in_window), default=0.0)

        model = pl.LpProblem("RollingMILP_Pred", pl.LpMinimize)
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

        pred_cost = pl.lpSum(
            self.pred_weight
            * sum(forecast[window[j][1].buyer_id][r] / self.world.capacity[(s, r)] for r in RESOURCES)
            * y[(j, s)]
            for j in range(len(window)) for s in self.world.sellers
        )

        model += (
            pl.lpSum(
                (ALPHA * self.world.latency_for_buyer(window[j][1].buyer_id)[s] + BETA * self.world.carbon[s])
                * y[(j, s)]
                for j in range(len(window)) for s in self.world.sellers
            )
            + pred_cost
            + pl.lpSum(REJECTION_PENALTY * (1 - z[j]) for j in range(len(window)))
        )

        self.log.info(
            f"SOLVING window_id={window_id} reason={reason} size={len(window)} "
            f"committing={commit_n} members={member_ids} max_forecast_cpu={max_forecast_cpu:.2f}"
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
                    f"REJECTED demand_id={it.demand_id} buyer={it.buyer_id} demand={d} in {window_id}"
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

            # EMA/history update happens regardless of accept/reject -- we still learned
            # this buyer's demand shape.
            for r in RESOURCES:
                self.ema_demand[it.buyer_id][r] = (
                    self.ema_alpha * d[r] + (1 - self.ema_alpha) * self.ema_demand[it.buyer_id][r]
                )
            self.history_count[it.buyer_id] += 1

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

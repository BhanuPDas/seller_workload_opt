"""
Shared helpers for the 4 algorithm processors: PuLP solve-status helpers,
the cost function (lifted straight from the validated v2 simulation --
ALPHA*latency + BETA*carbon), and the live-world resync logic each
processor uses now that capacity comes from real sellers instead of a
fixed simulated snapshot.
"""
import pulp as pl

from seller_workload_opt.v3.common.config import ALPHA, BETA, RESOURCES


def binval(var) -> float:
    v = var.value()
    return 0.0 if v is None else v


def solved_ok(model) -> bool:
    return pl.LpStatus[model.status] == "Optimal"


def safe_ratio(numerator: float, denominator: float) -> float:
    """
    numerator/denominator, but 0.0 when denominator is 0 instead of
    raising. Real sellers can legitimately report 0 capacity for a
    resource they don't offer (e.g. gpu=0 on a CPU-only box) -- the
    simulated world's randint(1, N) capacities never produced a zero, so
    this division-by-zero case didn't exist until now. The only case
    this is ever actually reached with a nonzero numerator is already
    infeasible anyway (a seller with 0 capacity for a resource a buyer
    needs is excluded by the capacity constraint/feasibility check
    before or alongside this), so returning 0.0 here doesn't hide a real
    contribution -- it just avoids crashing on it.
    """
    if not denominator:
        return 0.0
    return numerator / denominator


def resync_remaining(remaining: dict, last_synced: dict, world) -> None:
    """
    Reconciles a processor's private `remaining` dict against the live
    SellerWorld, per the "seller's next update overwrites ours" policy
    (see common/world.py's module docstring for the full reasoning):

    - A seller whose telemetry timestamp is newer than what this
      processor last incorporated gets its `remaining` snapped to the
      freshly reported capacity outright (our own in-between decrements
      for that seller are discarded -- the fresh number is ground truth).
    - A seller this processor has never seen gets added.
    - A seller no longer in the live world (stale/offline -- see
      SELLER_STALE_AFTER_SECONDS) gets removed from consideration.
    - A seller we've already synced at its current timestamp is left
      alone, so a processor's own commits since that refresh still count
      against it until the next real update arrives.

    Mutates `remaining` and `last_synced` in place. Call this at the top
    of every decision (each PrimalDual step, each MILP window/batch solve)
    -- it's a cheap dict diff, not a network call (SellerWorld.snapshot()
    reads the process-local registry a background thread keeps current).
    """
    snap = world.snapshot()

    for sid, info in snap.items():
        if last_synced.get(sid) != info["last_seen"]:
            for r in RESOURCES:
                remaining[(sid, r)] = info["capacity"].get(r, 0.0)
            last_synced[sid] = info["last_seen"]

    for sid in list(last_synced.keys()):
        if sid not in snap:
            last_synced.pop(sid, None)
            for r in RESOURCES:
                remaining.pop((sid, r), None)


def cost_of(buyer_id: str, seller: str, world) -> tuple:
    """Returns (latency_cost, carbon_cost, total_cost) for buyer->seller."""
    lat = world.latency_for_buyer(buyer_id)[seller]
    carbon = world.carbon[seller]
    lat_cost = ALPHA * lat
    carbon_cost = BETA * carbon
    return lat_cost, carbon_cost, lat_cost + carbon_cost


def rejected_decision(window_id, window_members, status, solve_ms) -> dict:
    return {
        "seller": None,
        "rejected": True,
        "lat_cost": 0.0,
        "carbon_cost": 0.0,
        "total_cost": None,  # not float("inf") -- Infinity isn't valid JSON
        "solve_ms": solve_ms,
        "window_id": window_id,
        "status": status,
        "window_members": window_members,
    }

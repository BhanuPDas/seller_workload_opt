"""
Shared helpers for the 4 algorithm processors: PuLP solve-status helpers
and the cost function, both lifted straight from the validated v2
simulation (ALPHA*latency + BETA*carbon).
"""
import pulp as pl

from seller_workload_opt.v3.common.config import ALPHA, BETA


def binval(var) -> float:
    v = var.value()
    return 0.0 if v is None else v


def solved_ok(model) -> bool:
    return pl.LpStatus[model.status] == "Optimal"


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

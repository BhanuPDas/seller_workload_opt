"""
Read/write helpers for per-algorithm decisions and the cross-algorithm
comparison result. Storage layout in Redis:

  decision:{demand_id}          HASH, one field per algorithm, each a
                                  JSON blob: {seller, rejected, lat_cost,
                                  carbon_cost, total_cost, solve_ms,
                                  window_id, status, window_members, ts}.
                                  (Not flattened with a "{algo}_" prefix:
                                  algorithm names here are prefixes of
                                  each other -- "rolling_milp" is a
                                  prefix of "rolling_milp_pred" -- so a
                                  flattened startswith() lookup silently
                                  cross-contaminates. One JSON field per
                                  algorithm avoids that entirely.)
  decisions-stream               STREAM, one event per algorithm decision
                                  (demand_id, algorithm, seller, rejected)
                                  -- what the comparator consumes.
  comparison:{demand_id}         HASH, written once all 4 algorithms have
                                  reported: recommended_algorithm,
                                  recommended_seller, agreement, plus a
                                  per-algorithm JSON recap, completed_ts.
"""
import json
import time

from seller_workload_opt.v3.common.config import ALGORITHMS, DECISION_KEY_TTL_SECONDS, DECISIONS_STREAM


def decision_key(demand_id: str) -> str:
    return f"decision:{demand_id}"


def comparison_key(demand_id: str) -> str:
    return f"comparison:{demand_id}"


def write_decision(r, algo_name: str, item, decision: dict, logger=None) -> None:
    key = decision_key(item.demand_id)
    record = {
        "seller": decision.get("seller") or "",
        "rejected": bool(decision.get("rejected")),
        "lat_cost": decision.get("lat_cost", 0.0),
        "carbon_cost": decision.get("carbon_cost", 0.0),
        "total_cost": decision.get("total_cost", 0.0),
        "solve_ms": decision.get("solve_ms", 0.0),
        "window_id": decision.get("window_id", ""),
        "status": decision.get("status", ""),
        "window_members": decision.get("window_members", []),
        "ts": time.time(),
    }
    pipe = r.pipeline()
    pipe.hset(key, algo_name, json.dumps(record))
    pipe.expire(key, DECISION_KEY_TTL_SECONDS)
    pipe.xadd(
        DECISIONS_STREAM,
        {
            "demand_id": item.demand_id,
            "buyer_id": item.buyer_id,
            "algorithm": algo_name,
            "seller": record["seller"],
            "rejected": int(record["rejected"]),
            # Redis stream fields can't hold None -- use empty string, same
            # convention as `seller` above; readers already treat "" as "no value".
            "total_cost": "" if record["total_cost"] is None else record["total_cost"],
        },
    )
    pipe.execute()
    if logger:
        cost_str = "n/a" if record["total_cost"] is None else f"{record['total_cost']:.3f}"
        logger.info(
            f"WROTE decision demand_id={item.demand_id} algo={algo_name} "
            f"seller={record['seller'] or None} rejected={record['rejected']} "
            f"total_cost={cost_str} solve_ms={record['solve_ms']:.1f} "
            f"window_id={record['window_id']} status={record['status']}"
        )


def read_all_decisions(r, demand_id: str) -> dict:
    """Returns {algo_name: {field: value, ...}} for algos that have reported."""
    raw = r.hgetall(decision_key(demand_id))
    out = {}
    for algo in ALGORITHMS:
        blob = raw.get(algo)
        if blob:
            try:
                out[algo] = json.loads(blob)
            except (TypeError, ValueError):
                continue
    return out


def is_complete(decisions: dict) -> bool:
    return all(algo in decisions for algo in ALGORITHMS)


def compute_recommendation(decisions: dict) -> dict:
    """
    Picks the lowest-total_cost, non-rejected result among the algorithms
    that produced one. Mirrors the objective the simulation already
    scores by (ALPHA*latency + BETA*carbon [+ predictive term]).
    """
    candidates = []
    for algo, fields in decisions.items():
        if fields.get("rejected"):
            continue
        seller = fields.get("seller")
        if not seller:
            continue
        try:
            cost = float(fields.get("total_cost", "inf"))
        except (TypeError, ValueError):
            continue
        candidates.append((cost, algo, seller))

    sellers_chosen = {fields.get("seller") for fields in decisions.values() if fields.get("seller")}

    if not candidates:
        return {
            "recommended_algorithm": "",
            "recommended_seller": "",
            "agreement": False,
            "all_rejected": True,
        }

    candidates.sort(key=lambda t: t[0])
    best_cost, best_algo, best_seller = candidates[0]
    return {
        "recommended_algorithm": best_algo,
        "recommended_seller": best_seller,
        "recommended_cost": best_cost,
        "agreement": len(sellers_chosen) <= 1,
        "all_rejected": False,
    }


def write_comparison(r, demand_id: str, buyer_id: str, decisions: dict, recommendation: dict, logger=None) -> None:
    key = comparison_key(demand_id)
    mapping = {"buyer_id": buyer_id, "completed_ts": time.time()}
    mapping.update({k: str(v) for k, v in recommendation.items()})
    for algo, fields in decisions.items():
        mapping[algo] = json.dumps({
            "seller": fields.get("seller", ""),
            "total_cost": fields.get("total_cost", ""),
            "rejected": fields.get("rejected", False),
        })
    r.hset(key, mapping=mapping)
    r.expire(key, DECISION_KEY_TTL_SECONDS)
    if logger:
        per_algo_parts = []
        for algo, fields in decisions.items():
            seller_label = fields.get("seller") or "REJECTED"
            per_algo_parts.append(f"{algo}={seller_label}")
        per_algo_str = ", ".join(per_algo_parts)
        logger.info(
            f"COMPARISON demand_id={demand_id} buyer={buyer_id} "
            f"recommended_algorithm={recommendation.get('recommended_algorithm')} "
            f"recommended_seller={recommendation.get('recommended_seller')} "
            f"agreement={recommendation.get('agreement')} "
            f"per_algo=[{per_algo_str}]"
        )

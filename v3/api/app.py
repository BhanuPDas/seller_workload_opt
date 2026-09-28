"""
Ingestion API. Styled after the existing ODEN transaction-initiation
handler: validate the incoming demand payload, log it, and hand off a
canonical record -- except here "hand off" means XADD to a Redis Stream
that all 4 algorithm workers consume (via their own consumer groups),
rather than a direct seller-selection call. Nothing here talks to an
algorithm directly; this process's only job is validate -> assign an id
-> enqueue.
"""
import json
import time
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request

from seller_workload_opt.v3.common.config import DEMAND_STREAM, ALGORITHMS
from seller_workload_opt.v3.common.logging_setup import get_logger
from seller_workload_opt.v3.common.models import DemandItem
from seller_workload_opt.v3.common.redis_client import get_redis
from seller_workload_opt.v3.common.decisions import read_all_decisions, comparison_key
from seller_workload_opt.v3.common.world import get_world

app = Flask(__name__)
logger = get_logger("api")
r = get_redis()
world = get_world()

SEQ_COUNTER_KEY = "demand:arrival_seq_counter"


def build_error(message, code=400):
    return jsonify({"status": "error", "message": message}), code


@app.route("/health", methods=["GET"])
def health():
    try:
        r.ping()
        return jsonify({"status": "ok", "redis": "up"}), 200
    except Exception as exc:
        return jsonify({"status": "error", "redis": "down", "detail": str(exc)}), 503


@app.route("/submit_demand", methods=["POST"])
def submit_demand():
    """
    Expected payload (mirrors the shape of the existing buyer-demand
    handler, minus the fields specific to on-chain settlement):

    {
      "buyer_id": "B12",
      "app_type": "batch-inference",
      "ip": "10.0.4.7",
      "lease_duration": 3600,
      "resources": {
        "cpu": {"demand_per_unit": 8},
        "mem": {"demand_per_unit": 16},
        "gpu": {"demand_per_unit": 1},
        "storage": {"demand_per_unit": 50}
      }
    }
    """
    try:
        data = request.get_json(silent=True)
        logger.info(f"Received request: {data}")

        if not data:
            logger.info("Empty or malformed JSON received")
            return build_error("Invalid request received")

        buyer_id = data.get("buyer_id")
        app_type = data.get("app_type")
        ip = data.get("ip")
        lease_duration = data.get("lease_duration")
        resources = data.get("resources")

        if not buyer_id or not lease_duration or not resources:
            logger.info(f"Missing required fields (buyer_id/lease_duration/resources): {data}")
            return build_error("Invalid request received")

        active_resources = {
            k: v for k, v in resources.items() if v.get("demand_per_unit", 0) > 0
        }
        if not active_resources:
            logger.info(f"No active resource demands in request: {data}")
            return build_error("At least one resource must have demand_per_unit > 0")

        unknown = set(active_resources) - {"cpu", "mem", "gpu", "storage"}
        if unknown:
            logger.info(f"Unsupported resource types requested: {unknown}")
            return build_error(f"Unsupported resource types: {sorted(unknown)}")

        demand_id = str(uuid.uuid4())
        arrival_seq = r.incr(SEQ_COUNTER_KEY)
        arrival_ts = time.time()

        item = DemandItem(
            demand_id=demand_id,
            buyer_id=buyer_id,
            app_type=app_type or "",
            lease_duration=float(lease_duration),
            cpu=float(active_resources.get("cpu", {}).get("demand_per_unit", 0)),
            mem=float(active_resources.get("mem", {}).get("demand_per_unit", 0)),
            gpu=float(active_resources.get("gpu", {}).get("demand_per_unit", 0)),
            storage=float(active_resources.get("storage", {}).get("demand_per_unit", 0)),
            arrival_seq=arrival_seq,
            arrival_ts=arrival_ts,
            ip=ip,
        )

        entry_id = r.xadd(DEMAND_STREAM, item.to_stream_fields())

        logger.info(
            f"ENQUEUED demand_id={demand_id} buyer={buyer_id} arrival_seq={arrival_seq} "
            f"stream_entry_id={entry_id} demand={item.demand()} "
            f"fanned out to algorithms={ALGORITHMS}"
        )

        return jsonify({
            "status": "queued",
            "demand_id": demand_id,
            "arrival_seq": arrival_seq,
            "submitted_at": datetime.fromtimestamp(arrival_ts, tz=timezone.utc).isoformat(),
            "poll_url": f"/decision/{demand_id}",
        }), 200

    except Exception as exc:
        logger.error(f"Unexpected error handling submit_demand: {exc}", exc_info=True)
        return jsonify({"status": "error", "message": "Internal error processing demand"}), 500


@app.route("/decision/<demand_id>", methods=["GET"])
def get_decision(demand_id):
    """
    Poll endpoint: returns whatever each of the 4 algorithms has decided
    so far for this demand_id, plus the cross-algorithm comparison once
    all 4 have reported. Batch-style algorithms may not have a result
    yet if their window/batch hasn't closed -- that's expected, not an
    error; keep polling.
    """
    decisions = read_all_decisions(r, demand_id)
    raw_comparison = r.hgetall(comparison_key(demand_id))

    comparison = None
    if raw_comparison:
        comparison = {}
        for k, v in raw_comparison.items():
            if k in ALGORITHMS:
                try:
                    comparison[k] = json.loads(v)
                    continue
                except (TypeError, ValueError):
                    pass
            comparison[k] = v

    reported = list(decisions.keys())
    pending = [a for a in ALGORITHMS if a not in decisions]

    return jsonify({
        "demand_id": demand_id,
        "reported": reported,
        "pending": pending,
        "decisions": decisions,
        "comparison": comparison or None,
    }), 200


@app.route("/sellers", methods=["GET"])
def list_sellers():
    """
    Debug endpoint: what this process's SellerWorld currently believes
    about the seller pool, straight off the live registry (not cached).
    Useful for confirming sellers are actually reaching the stream before
    chasing "everything gets rejected" through the algorithm logs.
    """
    snapshot = world.snapshot()
    now = time.time()
    sellers = {
        sid: {
            "capacity": info["capacity"],
            "carbon": info["carbon"],
            "seconds_since_last_update": round(now - info["last_seen"], 1),
            "collected_at": info.get("collected_at"),
        }
        for sid, info in snapshot.items()
    }
    return jsonify({"count": len(sellers), "sellers": sellers}), 200


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "marketplace-allocator-api",
        "algorithms": ALGORITHMS,
        "endpoints": [
            "/submit_demand [POST]", "/decision/<demand_id> [GET]",
            "/sellers [GET]", "/health [GET]",
        ],
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5801)

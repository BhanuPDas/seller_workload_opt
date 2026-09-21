"""
Comparator -- consumes decisions-stream (one event per algorithm decision)
and, once all 4 algorithms have reported for a given demand_id, computes
and logs which one is "most optimal" (lowest ALPHA*latency + BETA*carbon
among non-rejected results) plus whether all 4 agreed on the same seller.

Safe to run as exactly one instance: it only reads/aggregates, it never
touches seller capacity, so there's no shared-state race to worry about
here even though the 4 algorithm workers are fully independent processes.
"""
import time

from seller_workload_opt.v3.common.config import DECISIONS_STREAM, ALGORITHMS, COMPARATOR_WAIT_TIMEOUT_SECONDS
from seller_workload_opt.v3.common.decisions import read_all_decisions, is_complete, compute_recommendation, write_comparison
from seller_workload_opt.v3.common.logging_setup import get_logger
from seller_workload_opt.v3.common.redis_client import get_redis, ensure_consumer_group

GROUP = "cg-comparator"
CONSUMER = "comparator-0"


def run_comparator():
    logger = get_logger("comparator")
    r = get_redis()
    ensure_consumer_group(r, DECISIONS_STREAM, GROUP, logger=logger)

    logger.info(f"Comparator started: watching '{DECISIONS_STREAM}' for algorithms={ALGORITHMS}")

    pending_since = {}   # demand_id -> first-seen ts, cleared once compared
    already_compared = set()
    last_stale_check = time.time()

    while True:
        try:
            resp = r.xreadgroup(GROUP, CONSUMER, {DECISIONS_STREAM: ">"}, count=50, block=2000)
        except Exception as exc:
            logger.error(f"xreadgroup failed, retrying in 2s: {exc}", exc_info=True)
            time.sleep(2)
            continue

        if resp:
            for _, messages in resp:
                for msg_id, fields in messages:
                    demand_id = fields.get("demand_id")
                    buyer_id = fields.get("buyer_id", "")
                    algo = fields.get("algorithm")
                    logger.info(
                        f"observed demand_id={demand_id} buyer={buyer_id} algo={algo} "
                        f"seller={fields.get('seller') or 'REJECTED'}"
                    )

                    if demand_id not in pending_since and demand_id not in already_compared:
                        pending_since[demand_id] = time.time()

                    if demand_id not in already_compared:
                        decisions = read_all_decisions(r, demand_id)
                        if is_complete(decisions):
                            recommendation = compute_recommendation(decisions)
                            write_comparison(r, demand_id, buyer_id, decisions, recommendation, logger=logger)
                            already_compared.add(demand_id)
                            pending_since.pop(demand_id, None)
                        else:
                            missing = [a for a in ALGORITHMS if a not in decisions]
                            logger.debug(f"demand_id={demand_id} still waiting on {missing}")

                    r.xack(DECISIONS_STREAM, GROUP, msg_id)

        if time.time() - last_stale_check > 15:
            now = time.time()
            for demand_id, first_seen in list(pending_since.items()):
                waited = now - first_seen
                if waited > COMPARATOR_WAIT_TIMEOUT_SECONDS:
                    decisions = read_all_decisions(r, demand_id)
                    missing = [a for a in ALGORITHMS if a not in decisions]
                    logger.warning(
                        f"demand_id={demand_id} has been waiting {waited:.1f}s (> "
                        f"{COMPARATOR_WAIT_TIMEOUT_SECONDS}s) for algorithms={missing} to report -- "
                        f"likely still buffered in a window/batch that hasn't closed"
                    )
            last_stale_check = now


if __name__ == "__main__":
    run_comparator()

"""
Generic worker loop shared by all 4 algorithm workers. Each algorithm
gets its own Redis Stream consumer group on the same `demand-stream`, so
every group sees every demand item -- that's what gives each algorithm
an identical view of arrival order without the algorithms coordinating
with each other directly.

Ack discipline: a message is only XACK'd once its algorithm has actually
committed a decision for it (which for PrimalDual is immediate, but for
the 3 MILP-based algorithms can be several messages later, once their
window/batch closes). Until then it sits in the consumer group's
Pending Entries List, so a worker crash mid-window doesn't silently lose
that item -- XAUTOCLAIM on startup reclaims anything left pending by a
previous instance of the same algorithm.
"""
import os
import signal
import socket
import time

from seller_workload_opt.v3.common.config import (
    DEMAND_STREAM, XREAD_BLOCK_MS, XREAD_COUNT, PENDING_CLAIM_IDLE_MS,
)
from seller_workload_opt.v3.common.decisions import write_decision
from seller_workload_opt.v3.common.logging_setup import get_logger
from seller_workload_opt.v3.common.models import DemandItem
from seller_workload_opt.v3.common.redis_client import get_redis, ensure_consumer_group
from seller_workload_opt.v3.common.world import get_world
from seller_workload_opt.v3.algorithms.factory import build_processor

_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True


def run_worker(algo_name: str):
    logger = get_logger(algo_name)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    r = get_redis()
    group = f"cg-{algo_name}"
    consumer_name = f"{algo_name}-{socket.gethostname()}-{os.getpid()}"

    ensure_consumer_group(r, DEMAND_STREAM, group, logger=logger)

    world = get_world()
    processor = build_processor(algo_name, world, logger)

    logger.info(
        f"Worker started: algorithm={algo_name} consumer={consumer_name} "
        f"group={group} stream={DEMAND_STREAM}"
    )

    # Reclaim anything left pending by a previous (e.g. crashed) instance
    # of this same algorithm's consumer group before joining the live tail.
    _reclaim_pending(r, group, consumer_name, processor, logger)

    last_heartbeat = time.time()

    while not _shutdown_requested:
        try:
            resp = r.xreadgroup(
                groupname=group,
                consumername=consumer_name,
                streams={DEMAND_STREAM: ">"},
                count=XREAD_COUNT,
                block=XREAD_BLOCK_MS,
            )
        except Exception as exc:
            logger.error(f"xreadgroup failed, retrying in 2s: {exc}", exc_info=True)
            time.sleep(2)
            continue

        results = []
        if resp:
            for _, messages in resp:
                for msg_id, fields in messages:
                    try:
                        item = DemandItem.from_stream_fields(fields)
                    except Exception as exc:
                        logger.error(f"Malformed stream entry {msg_id}, acking to skip it: {exc}")
                        r.xack(DEMAND_STREAM, group, msg_id)
                        continue
                    logger.debug(f"RECEIVED demand_id={item.demand_id} msg_id={msg_id}")
                    results.extend(processor.step(msg_id, item))

        # Always give the processor a chance to flush on a time-based
        # trigger (max-wait timeout), even if no new messages arrived --
        # this is what stops a rolling/batch window stalling forever
        # during a quiet period.
        results.extend(processor.tick())

        for msg_id, item, decision in results:
            write_decision(r, algo_name, item, decision, logger=logger)
            r.xack(DEMAND_STREAM, group, msg_id)

        if time.time() - last_heartbeat > 30:
            logger.info(
                f"heartbeat: algorithm={algo_name} alive, "
                f"buffer_size={getattr(processor, 'buffer', None) and len(processor.buffer)}"
            )
            last_heartbeat = time.time()

    logger.info(f"Shutdown signal received, exiting cleanly: algorithm={algo_name}")


def _reclaim_pending(r, group, consumer_name, processor, logger):
    """
    XAUTOCLAIM any messages idle for longer than PENDING_CLAIM_IDLE_MS in
    this group (left over from a crashed/replaced worker instance) and
    feed them back through the processor before joining live traffic.
    """
    cursor = "0-0"
    reclaimed = 0
    while True:
        try:
            cursor, messages, _ = r.xautoclaim(
                DEMAND_STREAM, group, consumer_name, PENDING_CLAIM_IDLE_MS, cursor, count=50,
            )
        except Exception as exc:
            logger.warning(f"xautoclaim not available/failed (fine on a fresh stream): {exc}")
            return

        if not messages:
            break
        for msg_id, fields in messages:
            try:
                item = DemandItem.from_stream_fields(fields)
            except Exception:
                r.xack(DEMAND_STREAM, group, msg_id)
                continue
            reclaimed += 1
            logger.warning(
                f"RECLAIMED pending demand_id={item.demand_id} msg_id={msg_id} "
                f"(previously undelivered/uncommitted by another consumer)"
            )
            results = processor.step(msg_id, item)
            for r_msg_id, r_item, r_decision in results:
                write_decision(r, processor.name, r_item, r_decision, logger=logger)
                r.xack(DEMAND_STREAM, group, r_msg_id)
        if cursor == "0-0":
            break

    if reclaimed:
        logger.info(f"Reclaimed and re-fed {reclaimed} pending message(s) on startup")

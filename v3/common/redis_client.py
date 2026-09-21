import redis

from seller_workload_opt.v3.common.config import REDIS_HOST, REDIS_PORT, REDIS_DB


def get_redis() -> redis.Redis:
    """
    One connection factory used everywhere (API, workers, comparator).
    decode_responses=True so we work with plain str everywhere instead of
    bytes, which keeps the worker/algorithm code and log lines readable.
    """
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
    )


def ensure_consumer_group(r: redis.Redis, stream: str, group: str, logger=None) -> None:
    """
    Idempotently create a consumer group starting from the beginning of the
    stream (id='0'), creating the stream itself if it doesn't exist yet
    (MKSTREAM). Safe to call on every worker startup.
    """
    try:
        r.xgroup_create(name=stream, groupname=group, id="0", mkstream=True)
        if logger:
            logger.info(f"Created consumer group '{group}' on stream '{stream}'")
    except redis.exceptions.ResponseError as exc:
        if "BUSYGROUP" in str(exc):
            if logger:
                logger.info(f"Consumer group '{group}' already exists on '{stream}'")
        else:
            raise

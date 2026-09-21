"""
Structured, elapsed-time-stamped logging -- same spirit as the original
simulation script's log()/log_solve()/log_trace() helpers, but using the
stdlib `logging` module so it plays nicely with `docker compose logs -f
<service>` and can be redirected/aggregated later without code changes.

Every log line is prefixed with the algorithm/service name (via the
logger name) and carries elapsed-since-process-start seconds, matching
the debugging style already validated in the simulation.
"""
import logging
import sys
import time

from seller_workload_opt.v3.common.config import LOG_LEVEL

_PROC_START = time.time()


class ElapsedFormatter(logging.Formatter):
    def format(self, record):
        record.elapsed = f"{time.time() - _PROC_START:8.2f}s"
        return super().format(record)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (avoid duplicate handlers)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        ElapsedFormatter(
            fmt="[%(elapsed)s][%(levelname)-5s][%(name)s] %(message)s"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    logger.propagate = False
    return logger

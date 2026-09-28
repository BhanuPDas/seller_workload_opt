"""
Central configuration, entirely env-var driven so the same Docker image
can run the API, any of the 4 algorithm workers, or the comparator just
by changing environment variables (see docker-compose.yml).
"""
import os


def _int(name, default):
    return int(os.environ.get(name, default))


def _float(name, default):
    return float(os.environ.get(name, default))


# --- Redis ---------------------------------------------------------------
REDIS_HOST = os.environ.get("REDIS_HOST", "redis")
REDIS_PORT = _int("REDIS_PORT", 6379)
REDIS_DB = _int("REDIS_DB", 0)

DEMAND_STREAM = os.environ.get("DEMAND_STREAM", "demand-stream")
DECISIONS_STREAM = os.environ.get("DECISIONS_STREAM", "decisions-stream")
DECISION_KEY_TTL_SECONDS = _int("DECISION_KEY_TTL_SECONDS", 7 * 24 * 3600)

# --- Live seller world (real sellers publish here -- see common/world.py) --
# Every process that needs seller state (the API + all 4 workers) tails
# this stream independently with a plain XREAD (no consumer group --
# this is reference data everyone needs a full copy of, not a work queue
# to fan out). Each stream entry is one seller's current reported state;
# see common/world.py's module docstring for the expected field shape
# and how to adapt it to your real message format.
SELLER_STREAM = os.environ.get("SELLER_STREAM", "seller-updates")
# A seller not heard from in this long is dropped from the active pool
# (treated as offline) rather than kept around on stale numbers forever.
# Set to 0 to disable expiry entirely.
SELLER_STALE_AFTER_SECONDS = _int("SELLER_STALE_AFTER_SECONDS", 180)
# Fallback values used only when a seller's message omits that field.
DEFAULT_CARBON = _float("DEFAULT_CARBON", 5.0)
# No real buyer<->seller network/geo model yet -- every seller gets this
# flat latency unless/until real location data is being published (see
# world.py's _estimate_latency()).
DEFAULT_LATENCY = _float("DEFAULT_LATENCY", 10.0)

# --- Objective weights (same as the validated simulation) ----------------
ALPHA = _float("ALPHA", 1.0)
BETA = _float("BETA", 0.3)
REJECTION_PENALTY = _float("REJECTION_PENALTY", 100.0)
RESOURCES = ["cpu", "mem", "gpu", "storage"]
GAMMA_SCARCITY = {
    "cpu": _float("GAMMA_SCARCITY_CPU", 0.5),
    "mem": _float("GAMMA_SCARCITY_MEM", 0.15),
    "gpu": _float("GAMMA_SCARCITY_GPU", 1.0),
    "storage": _float("GAMMA_SCARCITY_STORAGE", 0.1),
}

# --- RollingMILP+ ----------------------------------------------------------
ROLLING_K = _int("ROLLING_K", 20)
ROLLING_S = _int("ROLLING_S", 5)
ROLLING_MAX_WAIT_SECONDS = _float("ROLLING_MAX_WAIT_SECONDS", 8.0)
ROLLING_SOLVE_TIME_LIMIT = _int("ROLLING_SOLVE_TIME_LIMIT", 10)

# --- RollingMILPPred -------------------------------------------------------
PRED_K = _int("PRED_K", 20)
PRED_S = _int("PRED_S", 5)
PRED_MAX_WAIT_SECONDS = _float("PRED_MAX_WAIT_SECONDS", 8.0)
PRED_SOLVE_TIME_LIMIT = _int("PRED_SOLVE_TIME_LIMIT", 10)
PRED_W = _int("PRED_W", 20)          # confidence ramp-up window (# obs)
PRED_F = _int("PRED_F", 2)           # forecast horizon multiplier
PRED_W_MIN = _int("PRED_W_MIN", 5)   # min obs before forecasting at all
PRED_WEIGHT = _float("PRED_WEIGHT", 0.5)
PRED_EMA_ALPHA = _float("PRED_EMA_ALPHA", 0.5)

# --- BatchMILP+ --------------------------------------------------------
BATCH_SIZE = _int("BATCH_SIZE", 10)
BATCH_MAX_WAIT_SECONDS = _float("BATCH_MAX_WAIT_SECONDS", 10.0)
BATCH_SOLVE_TIME_LIMIT = _int("BATCH_SOLVE_TIME_LIMIT", 10)

# --- PrimalDual ----------------------------------------------------------
PD_ETA = _float("PD_ETA", 0.5)
PD_DECAY = _float("PD_DECAY", 0.01)
PD_GAMMA_S = _float("PD_GAMMA_S", 2.0)

# --- Worker loop tuning ----------------------------------------------------
XREAD_BLOCK_MS = _int("XREAD_BLOCK_MS", 2000)
XREAD_COUNT = _int("XREAD_COUNT", 20)
PENDING_CLAIM_IDLE_MS = _int("PENDING_CLAIM_IDLE_MS", 30000)

# --- Comparator ------------------------------------------------------------
COMPARATOR_WAIT_TIMEOUT_SECONDS = _float("COMPARATOR_WAIT_TIMEOUT_SECONDS", 60.0)
COMPARATOR_POLL_SECONDS = _float("COMPARATOR_POLL_SECONDS", 1.0)

ALGORITHMS = ["rolling_milp", "rolling_milp_pred", "batch_milp", "primal_dual"]

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

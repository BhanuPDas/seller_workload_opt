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
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = _int("REDIS_PORT", 6379)
REDIS_DB = _int("REDIS_DB", 0)

DEMAND_STREAM = os.environ.get("DEMAND_STREAM", "demand-stream")
DECISIONS_STREAM = os.environ.get("DECISIONS_STREAM", "decisions-stream")
DECISION_KEY_TTL_SECONDS = _int("DECISION_KEY_TTL_SECONDS", 7 * 24 * 3600)

# --- Simulated world (stand-in for real seller discovery/capacity) -------
# IMPORTANT: this seed must be identical across the API and all 4 workers
# so every algorithm is scored against an *identical* initial world. Each
# worker still keeps its own private, independently-mutated copy of
# `remaining capacity` after that -- see common/world.py.
WORLD_SEED = _int("WORLD_SEED", 42)
NUM_SELLERS = _int("NUM_SELLERS", 48)
BUYER_DEGREE = _int("BUYER_DEGREE", 5)  # sellers each buyer is "wired" to

# --- Objective weights (same as the validated simulation) ----------------
ALPHA = _float("ALPHA", 1.0)
BETA = _float("BETA", 0.3)
REJECTION_PENALTY = _float("REJECTION_PENALTY", 100.0)
RESOURCES = ["cpu", "mem", "gpu"]
GAMMA_SCARCITY = {
    "cpu": _float("GAMMA_SCARCITY_CPU", 0.5),
    "mem": _float("GAMMA_SCARCITY_MEM", 0.15),
    "gpu": _float("GAMMA_SCARCITY_GPU", 1.0),
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

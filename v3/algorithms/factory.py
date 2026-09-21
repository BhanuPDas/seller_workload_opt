from seller_workload_opt.v3.algorithms.rolling_milp import RollingMilpProcessor
from seller_workload_opt.v3.algorithms.rolling_milp_pred import RollingMilpPredProcessor
from seller_workload_opt.v3.algorithms.batch_milp import BatchMilpProcessor
from seller_workload_opt.v3.algorithms.primal_dual import PrimalDualProcessor

PROCESSORS = {
    "rolling_milp": RollingMilpProcessor,
    "rolling_milp_pred": RollingMilpPredProcessor,
    "batch_milp": BatchMilpProcessor,
    "primal_dual": PrimalDualProcessor,
}


def build_processor(algo_name, world, logger):
    if algo_name not in PROCESSORS:
        raise ValueError(f"Unknown algorithm '{algo_name}'. Known: {list(PROCESSORS)}")
    return PROCESSORS[algo_name](world, logger)

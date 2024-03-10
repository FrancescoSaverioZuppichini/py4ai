import json
import logging.config
import logging
from typing import Optional
from torch.profiler import profile, schedule, ProfilerActivity


def get_logger(logger_name: Optional[str] = None) -> logging.Logger:
    with open("logging_config.json", "r") as f:
        config = json.load(f)

    logging.config.dictConfig(config)
    return logging.getLogger(logger_name)


def get_torch_profiler() -> profile:
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # schedule=schedule(wait=0, warmup=8, repeat=4, active=1),
    )

import json
import logging
import logging.config
from typing import Optional

from torch.profiler import ProfilerActivity, profile, schedule


def get_logger(logger_name: Optional[str] = None) -> logging.Logger:
    with open("logging_config.json", "r") as f:
        config = json.load(f)

    logging.config.dictConfig(config)
    return logging.getLogger(logger_name)


def get_torch_profiler() -> profile:
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    )

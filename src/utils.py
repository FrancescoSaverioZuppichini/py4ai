import json 
import logging.config
import logging
from typing import Optional

def get_logger(logger_name: Optional[str] = None) -> logging.Logger:
    with open("logging_config.json", "r") as f:
        config = json.load(f)

    logging.config.dictConfig(config)
    return logging.getLogger(logger_name)


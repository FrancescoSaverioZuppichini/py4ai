from lib.utils import get_logger


logger = get_logger()

logger.info({"key": "foo", "batch_size": 8, "time": 200, "model": "a"})
logger.info({"key": "foo", "batch_size": 16, "time": 400, "model": "a"})
logger.info({"key": "foo", "batch_size": 8, "time": 150, "model": "b"})
logger.info({"key": "foo", "batch_size": 16, "time": 500, "model": "b"})

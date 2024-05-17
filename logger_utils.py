import logging
from typing import Optional
from logging import Logger
from typing import Callable
from functools import wraps
import time


def log_execution_time(logger: Logger, message: str) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            logger.debug(f"{message} - {execution_time:.4f} seconds")
            return result
        return wrapper
    return decorator


def get_logger(name: str, level: int, logs_path: Optional[str] = None) -> Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # fix duplicates
    logger.propagate = False
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if logs_path is not None:
        file_handler = logging.FileHandler(logs_path, encoding="utf-8")
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
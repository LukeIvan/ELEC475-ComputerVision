from loguru import logger
import os

class Logger:
    def __init__(self, log_dir="logs"):
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "training_log_{time:YYYY-MM-DD_HH-mm-ss}.log")
        logger.add(log_file, format="{time} {level} {message}")

    def info(self, msg):
        logger.info(msg)

    def error(self, msg):
        logger.error(msg)

    def warning(self, msg):
        logger.warning(msg)

    def debug(self, msg):
        logger.debug(msg)

    def success(self, msg):
        logger.success(msg)

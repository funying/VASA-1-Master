# utils/logger.py

import logging
import os

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup logger to log training progress."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

if __name__ == "__main__":
    logger = setup_logger('train', 'train.log')
    logger.info('This is a test log message.')

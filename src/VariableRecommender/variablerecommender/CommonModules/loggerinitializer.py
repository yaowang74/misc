# -*- coding: utf-8 -*-
"""
Logger

@author: H211803
"""

import logging
import sys
import os.path


def InitializeLogger(app_name):
    """initialize a logger

    """
    log_output_dir = os.path.dirname(os.path.abspath(__file__))
    logger = logging.getLogger(app_name)
    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
            fmt='%(asctime)s.%(msecs)03d %(name)s:%(levelname)s: %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create error file handler and set level to error
    handler = logging.FileHandler(os.path.join(log_output_dir,
                                               "error.log"), "w",
                                  encoding=None, delay="true")
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter(
            fmt='%(asctime)s.%(msecs)03d %(name)s:%(levelname)s: %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler(os.path.join(log_output_dir, "all.log"), "w")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
            fmt='%(asctime)s.%(msecs)03d %(name)s:%(levelname)s: %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

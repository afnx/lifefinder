"""
Logging utility for the LifeFinder project.
"""

import logging
import sys


def get_logger(name=__name__, level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(name)

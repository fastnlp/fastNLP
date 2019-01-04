import logging
import os


def create_logger(logger_name, log_path, log_format=None, log_level=logging.INFO):
    """Create a logger.

    :param str logger_name:
    :param str log_path:
    :param log_format:
    :param log_level:
    :return: logger

    To use a logger::

        logger.debug("this is a debug message")
        logger.info("this is a info message")
        logger.warning("this is a warning message")
        logger.error("this is an error message")
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    if log_path is None:
        handler = logging.StreamHandler()
    else:
        os.stat(os.path.dirname(os.path.abspath(log_path)))
        handler = logging.FileHandler(log_path)
    handler.setLevel(log_level)
    if log_format is None:
        log_format = "[%(asctime)s %(name)-13s %(levelname)s %(process)d %(thread)d " \
                     "%(filename)s:%(lineno)-5d] %(message)s"
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

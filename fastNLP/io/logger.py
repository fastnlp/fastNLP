import logging
import logging.config
import torch
import _pickle as pickle
import os
import sys
import warnings


__all__  = ['logger']

try:
    import fitlog
except ImportError:
    fitlog = None
try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

if tqdm is not None:
    class TqdmLoggingHandler(logging.Handler):
        def __init__(self, level=logging.INFO):
            super().__init__(level)

        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)
else:
    class TqdmLoggingHandler(logging.StreamHandler):
        def __init__(self, level=logging.INFO):
            super().__init__(sys.stdout)
            self.setLevel(level)


def get_level(level):
    if isinstance(level, int):
        pass
    else:
        level = level.lower()
        level = {'info': logging.INFO, 'debug': logging.DEBUG,
                 'warn': logging.WARN, 'warning': logging.WARN,
                 'error': logging.ERROR}[level]
    return level


def init_logger(path=None, stdout='tqdm', level='INFO'):
    """initialize logger"""
    if stdout not in ['none', 'plain', 'tqdm']:
        raise ValueError('stdout must in one of {}'.format(['none', 'plain', 'tqdm']))

    level = get_level(level)

    logger = logging.getLogger('fastNLP')
    logger.setLevel(level)
    handlers_type = set([type(h) for h in logger.handlers])

    # make sure to initialize logger only once
    # Stream Handler
    if stdout == 'plain' and (logging.StreamHandler not in handlers_type):
        stream_handler = logging.StreamHandler(sys.stdout)
    elif stdout == 'tqdm' and (TqdmLoggingHandler not in handlers_type):
        stream_handler = TqdmLoggingHandler(level)
    else:
        stream_handler = None

    if stream_handler is not None:
        stream_formatter = logging.Formatter('[%(levelname)s] %(message)s')
        stream_handler.setLevel(level)
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    # File Handler
    if path is not None and (logging.FileHandler not in handlers_type):
        if os.path.exists(path):
            assert os.path.isfile(path)
            warnings.warn('log already exists in {}'.format(path))
        dirname = os.path.abspath(os.path.dirname(path))
        os.makedirs(dirname, exist_ok=True)

        file_handler = logging.FileHandler(path, mode='a')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(fmt='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
                                           datefmt='%Y/%m/%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger


# init logger when import
logger = init_logger()


def get_logger(name=None):
    if name is None:
        return logging.getLogger('fastNLP')
    return logging.getLogger(name)


def set_file(path, level='INFO'):
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            if os.path.abspath(path) == h.baseFilename:
                # file path already added
                return

    # File Handler
    if os.path.exists(path):
        assert os.path.isfile(path)
        warnings.warn('log already exists in {}'.format(path))
    dirname = os.path.abspath(os.path.dirname(path))
    os.makedirs(dirname, exist_ok=True)

    file_handler = logging.FileHandler(path, mode='a')
    file_handler.setLevel(get_level(level))
    file_formatter = logging.Formatter(fmt='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
                                       datefmt='%Y/%m/%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


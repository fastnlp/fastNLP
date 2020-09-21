r"""
Logger 是fastNLP中记录日志的模块，logger封装了logging模块的Logger，
具体使用方式与直接使用logging.Logger相同，同时也新增一些简单好用的API
使用方式：
from fastNLP import logger
#
# logger 可以和 logging.Logger 一样使用
logger.info('your msg')
logger.error('your msg')

# logger 新增的API
# 将日志输出到文件，以及输出的日志等级
logger.add_file('/path/to/log', level='INFO')
# 定义在命令行中的显示格式和日志等级
logger.set_stdout('tqdm', level='WARN')

"""

__all__ = [
    'logger',
    'init_logger_dist'
]

import logging
import logging.config
import os
import sys
import warnings
from torch import distributed as dist

ROOT_NAME = 'fastNLP'

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


def _get_level(level):
    if isinstance(level, int):
        pass
    else:
        level = level.lower()
        level = {'info': logging.INFO, 'debug': logging.DEBUG,
                 'warn': logging.WARN, 'warning': logging.WARN,
                 'error': logging.ERROR}[level]
    return level


def _add_file_handler(logger, path, level='INFO'):
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
    file_handler.setLevel(_get_level(level))
    file_formatter = logging.Formatter(fmt='%(asctime)s - %(module)s - [%(levelname)s] - %(message)s',
                                       datefmt='%Y/%m/%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def _set_stdout_handler(logger, stdout='tqdm', level='INFO'):
    level = _get_level(level)
    if stdout not in ['none', 'plain', 'tqdm']:
        raise ValueError('stdout must in one of {}'.format(['none', 'plain', 'tqdm']))
    # make sure to initialize logger only once
    stream_handler = None
    for i, h in enumerate(logger.handlers):
        if isinstance(h, (logging.StreamHandler, TqdmLoggingHandler)):
            stream_handler = h
            break
    if stream_handler is not None:
        logger.removeHandler(stream_handler)
    
    # Stream Handler
    if stdout == 'plain':
        stream_handler = logging.StreamHandler(sys.stdout)
    elif stdout == 'tqdm':
        stream_handler = TqdmLoggingHandler(level)
    else:
        stream_handler = None
    
    if stream_handler is not None:
        stream_formatter = logging.Formatter('%(message)s')
        stream_handler.setLevel(level)
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)


class FastNLPLogger(logging.getLoggerClass()):
    def __init__(self, name):
        super().__init__(name)
    
    def add_file(self, path='./log.txt', level='INFO'):
        r"""add log output file and the output level"""
        _add_file_handler(self, path, level)
    
    def set_stdout(self, stdout='tqdm', level='INFO'):
        r"""set stdout format and the output level"""
        _set_stdout_handler(self, stdout, level)


logging.setLoggerClass(FastNLPLogger)


# print(logging.getLoggerClass())
# print(logging.getLogger())

def _init_logger(path=None, stdout='tqdm', level='INFO'):
    r"""initialize logger"""
    level = _get_level(level)
    
    # logger = logging.getLogger()
    logger = logging.getLogger(ROOT_NAME)
    logger.propagate = False
    logger.setLevel(1)  # make the logger the lowest level
    
    _set_stdout_handler(logger, stdout, level)
    
    # File Handler
    if path is not None:
        _add_file_handler(logger, path, level)
    
    return logger


def _get_logger(name=None, level='INFO'):
    level = _get_level(level)
    if name is None:
        name = ROOT_NAME
    assert isinstance(name, str)
    if not name.startswith(ROOT_NAME):
        name = '{}.{}'.format(ROOT_NAME, name)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


logger = _init_logger(path=None, level='INFO')


def init_logger_dist():
    global logger
    rank = dist.get_rank()
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)

import time
from contextlib import contextmanager


@contextmanager
def check_time_elapse(seconds, op='lt'):
    """
    检测某一段程序所花费的时间，是否 op 给定的seconds

    :param int seconds:
    :param str op:
    :return:
    """
    start = time.time()
    yield
    end = time.time()
    if op == 'lt':
        assert end-start < seconds
    elif op == 'gt':
        assert end-start > seconds
    elif op == 'eq':
        assert end - start == seconds
    elif op == 'le':
        assert end - start <= seconds
    elif op == 'ge':
        assert end - start >= seconds
    else:
        raise ValueError("Only supports lt,gt,eq,le,ge.")






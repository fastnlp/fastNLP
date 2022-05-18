import pytest
from fastNLP.core.dataloaders.utils import _match_param
from fastNLP import logger
from tests.helpers.utils import recover_logger, Capturing


def demo():
    pass


def test_no_args():
    def f(*args, a, b, **kwarg):
        c = 100
        call_kwargs = _match_param(f, demo)
    f(a=1, b=2)

    def f(a, *args, b, **kwarg):
        c = 100
        call_kwargs = _match_param(f, demo)
    f(a=1, b=2)


@recover_logger
def test_warning():
    logger.set_stdout('raw')
    def f1(a, b):
        return 1

    def f2(a, b, c=2):
        kwargs = _match_param(f2, f1)
        return f1(*kwargs)

    with Capturing() as out:
        f2(a=1, b=2, c=3)
    assert 'Parameter:c' in out[0]  # 传入了需要 warning

    assert f2(1, 2) == 1

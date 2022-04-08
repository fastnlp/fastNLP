import os
from functools import wraps
from typing import Callable, Any, Optional
from contextlib import contextmanager

__all__ = [
    'is_cur_env_distributed',
    'get_global_rank',
    'rank_zero_call',
    'all_rank_call'
]

from fastNLP.envs.env import FASTNLP_GLOBAL_RANK


def is_cur_env_distributed() -> bool:
    """
    单卡模式该函数一定返回 False；
    注意进程 0 在多卡的训练模式下前后的值是不一样的，例如在开启多卡的 driver 之前，在进程 0 上的该函数返回 False；但是在开启后，在进程 0 上
     的该函数返回的值是 True；
    多卡模式下除了进程 0 外的其它进程返回的值一定是 True；
    """
    return FASTNLP_GLOBAL_RANK in os.environ


def get_global_rank():
    return int(os.environ.get(FASTNLP_GLOBAL_RANK, 0))


def rank_zero_call(fn: Callable):
    """
    通过该函数包裹的函数，在单卡模式下该方法不影响任何东西，在多卡状态下仅会在 global rank 为 0 的进程下执行。使用方式有两种

    # 使用方式1
        @rank_zero_call
        def save_model():
            do_something # will only run in global rank 0

    # 使用方式2
        def add(a, b):
            return a+b
        rank_zero_call(add)(1, 2)

    :param fn: 需要包裹的可执行的函数。
    :return:
    """
    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        if int(os.environ.get(FASTNLP_GLOBAL_RANK, 0)) == 0:
            return fn(*args, **kwargs)
        return None
    return wrapped_fn


@contextmanager
def all_rank_call():
    """
    在多卡模式下，该环境内，会暂时地将 FASTNLP_GLOBAL_RANK 设置为 "0"，使得 rank_zero_call 函数失效，使得每个进程都会运行该函数。

    # 使用方式
    with all_rank_run():
        do_something  # all rank will do

    :param fn:
    :return:
    """
    old_fastnlp_global_rank = os.environ[FASTNLP_GLOBAL_RANK] if FASTNLP_GLOBAL_RANK in os.environ else None
    os.environ[FASTNLP_GLOBAL_RANK] = '0'

    yield

    if old_fastnlp_global_rank is not None:
        os.environ[FASTNLP_GLOBAL_RANK] = old_fastnlp_global_rank
    else:
        os.environ.pop(FASTNLP_GLOBAL_RANK)

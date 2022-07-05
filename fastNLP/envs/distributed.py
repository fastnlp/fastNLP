import os
from functools import wraps
from pathlib import Path
from typing import Callable, Any, Optional, Union
from contextlib import contextmanager

__all__ = [
    'is_cur_env_distributed',
    'get_global_rank',
    'rank_zero_call',
    'all_rank_call_context',
    'fastnlp_no_sync_context',
    "rank_zero_rm"
]

from fastNLP.envs.env import FASTNLP_GLOBAL_RANK, FASTNLP_NO_SYNC


def is_cur_env_distributed() -> bool:
    """
    判断当前是否处于分布式的环境下。单卡模式该函数一定返回 ``False``；
    注意进程 0 在多卡的训练模式下前后的值是不一样的，例如在开启多卡的 driver 之前，在进程 0 上的该函数返回 ``False`` ；但是在开启后，在进程 0 上
    的该函数返回的值是 ``True`` ；多卡模式下除了进程 0 外的其它进程返回的值一定是 ``True`` 。
    """
    return FASTNLP_GLOBAL_RANK in os.environ


def get_global_rank()->int:
    """
    获取当前进程的 global_rank 。

    :return:
    """
    return int(os.environ.get(FASTNLP_GLOBAL_RANK, 0))


def rank_zero_call(fn: Callable):
    """
    通过该函数包裹的函数，在单卡模式下该方法不影响任何东西，在多卡状态下仅会在 global rank 为 0 的进程下执行。使用方式有两种

    使用方式1::

        @rank_zero_call
        def save_model():
            do_something # will only run in global rank 0

    使用方式2::

        def add(a, b):
            return a+b
        rank_zero_call(add)(1, 2)

    同时，该函数还会设置环境变量 ``FASTNLP_NO_SYNC`` 为 **2** ，在这个环境下，所有的 **fastNLP** 内置的 :meth:`barrier` 接口和 ``gather`` / ``broadcast``
    操作都没有任何意义。

    :param fn: 需要包裹的可执行的函数。
    :return:
    """
    @wraps(fn)
    def wrapped_fn(*args: Any, **kwargs: Any) -> Optional[Any]:
        if int(os.environ.get(FASTNLP_GLOBAL_RANK, 0)) == 0:
            with fastnlp_no_sync_context(level=2):
                return fn(*args, **kwargs)
        return None
    return wrapped_fn


@contextmanager
def fastnlp_no_sync_context(level: int = 2):
    """
    用于让 **fastNLP** 的 :meth:`barrier` 以及 ``gather`` / ``broadcast`` 等操作等同于只有 1 卡的多卡程序。如果为 1 表示 **fastNLP** 里的
    :meth:`barrier` 操作失效；如果为 2 表示 :meth:`barrier` 与 ``gather`` / ``broadcast`` 都失效。

    :param level: 可选 ``[0, 1, 2]``
    """
    old_level = os.environ.get(FASTNLP_NO_SYNC, None)
    os.environ[FASTNLP_NO_SYNC] = f'{level}'
    yield
    if old_level is None:
        os.environ.pop(FASTNLP_NO_SYNC)
    else:
        os.environ[FASTNLP_NO_SYNC] = old_level


@contextmanager
def all_rank_call_context():
    """
    在多卡模式下，该环境内，会暂时地将 ``FASTNLP_GLOBAL_RAN``K 设置为 **"0"** ，使得 :func:`rank_zero_call` 函数失效，使得每个进程都会运行该函数。

    使用方式::

        with all_rank_call_context():
            do_something  # all rank will do

    """
    old_fastnlp_global_rank = os.environ[FASTNLP_GLOBAL_RANK] if FASTNLP_GLOBAL_RANK in os.environ else None
    os.environ[FASTNLP_GLOBAL_RANK] = '0'

    yield

    if old_fastnlp_global_rank is not None:
        os.environ[FASTNLP_GLOBAL_RANK] = old_fastnlp_global_rank
    else:
        os.environ.pop(FASTNLP_GLOBAL_RANK)


def rank_zero_rm(path: Optional[Union[str, Path]]):
    """
    仅在 rank 0 下删除文件的函数。普通的删除文件操作在分布式文件系统中可能会发生错误，rank 0 下发删除成功后就运行走了，但实际的删除需要 rank 0 的机器
    发送到远程文件系统再去执行，这个时候在 rank 0 已经删除成功了，但是在远程文件系统那里这个操作还没完成，rank 1 读取的时候还是读取到存在这个文件；
    该函数会保证所有进程都检测到 ``path`` 删除之后才退出，请保证不同进程上 ``path`` 是完全一样的，否则会陷入死锁状态。

    :param path:
    """
    if int(os.environ.get(FASTNLP_GLOBAL_RANK, 0)) == 0:
        if path is None:
            return
        if isinstance(path, str):
            path = Path(path)
        if not path.exists():
            return
        _recursive_rm(path)


def _recursive_rm(path: Path):
    if path.is_file() or path.is_symlink():
        if path.exists():
            try:
                path.unlink()
            except Exception:
                pass
        return
    for sub_path in list(path.iterdir()):
        _recursive_rm(sub_path)
    path.rmdir()
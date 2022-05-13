__all__ = [
    'prepare_dataloader'
]

from typing import Union, Callable
import os
import sys

from ..samplers import RandomBatchSampler, RandomSampler
from .torch_dataloader import prepare_torch_dataloader
from .paddle_dataloader import prepare_paddle_dataloader
from .jittor_dataloader import prepare_jittor_dataloader
from ...envs import FASTNLP_BACKEND, SUPPORT_BACKENDS
from ..log import logger


def prepare_dataloader(dataset, batch_size: int = 16, shuffle: bool = True, drop_last: bool = False,
             collate_fn: Union[Callable, str, None] = 'auto', num_workers: int = 0,
             seed: int = 0, backend: str = 'auto'):
    """
    自动创建合适的 ``DataLoader`` 对象。例如，检测当当前环境是 ``torch`` 的，则返回 ``TorchDataLoader`` , 是 ``paddle`` 的则
    返回 ``PaddleDataLoader`` 。如果有更多需要定制的参数，请直接使用对应的 ``prepare`` 函数，例如
     :func:`~fastNLP.prepare_torch_dataloader` 或  :func:`~fastNLP.prepare_paddle_dataloader` 等。

    :param dataset: 实现 __getitem__() 和 __len__() 的对象；或这种对象的序列；或字典。

        * 为单个数据集对象时
         返回一个 DataLoader 。
        * 为数据集对象序列时
         返回一个序列的 DataLoader 。
        * 为字典型 或 :class:`~fastNLP.io.DataBundle` 数据时，返回 `Dict` 类型的数据。
         返回一个字典 。

    :param batch_size: 批次大小。
    :param shuffle: 是否打乱数据集。
    :param drop_last: 当最后一个 batch 不足 batch_size 数量的是否，是否丢弃。
    :param collate_fn: 用于处理一个 batch 的函数，一般包括 padding 和转为 tensor。有以下三种取值：

        * 为 ``auto`` 时
         使用 :class:`~fastNLP.Collator` 进行 padding 和 转tensor 。
        * 为 ``Callable`` 时
         应当接受一个 ``batch`` 的数据作为参数，同时输出一个对象 。
        * 为 ``None`` 时
         使用各个框架的 DataLoader 的默认 ``collate_fn`` 。
    :param num_workers: 使用多少进程进行数据的 fetch 。
    :param seed: 使用的随机数种子。
    :param backend: 当前支持 ``["auto", "torch", "paddle", "jittor"]`` 四种类型。

        * 为 ``auto`` 时
         首先(1) 根据环境变量 "FASTNLP_BACKEND" 进行判断；如果没有设置则，(2）通过当前 ``sys.modules`` 中已经 import 的
          ``backend`` 进行判定。如果以上均无法判定，则报错。如果找到了 ``backend`` ，则按照下述的方式处理。
        * 为 ``torch`` 时
         使用 :func:`~fastNLP.prepare_torch_dataloader` 。
        * 为 ``paddle`` 时
         使用 :func:`~fastNLP.prepare_paddle_dataloader` 。
        * 为 ``jittor`` 时
         使用 :func:`~fastNLP.prepare_jittor_dataloader` 。

    :return
    """
    if backend == 'auto':
        backend = _get_backend()
    if backend == 'torch':
        batch_sampler = RandomBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                           drop_last=drop_last, seed=seed)
        return prepare_torch_dataloader(ds_or_db=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn,
                                        num_workers=num_workers, shuffle=False, sampler=None)
    elif backend == 'paddle':
        batch_sampler = RandomBatchSampler(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                                           drop_last=drop_last, seed=seed)
        return prepare_paddle_dataloader(ds_or_db=dataset, batch_sampler=batch_sampler, collate_fn=collate_fn,
                                        num_workers=num_workers)
    elif backend == 'jittor':
        sampler = RandomSampler(dataset=dataset, shuffle=shuffle, seed=seed)
        prepare_jittor_dataloader(ds_or_db=dataset, sampler=sampler, collate_fn=collate_fn,
                                  num_workers=num_workers, batch_size=batch_size, shuffle=shuffle,
                                  drop_last=drop_last)
    else:
        raise ValueError(f"Currently we do not support backend:{backend}.")


def _check_module(module):
    """
    检查该 module 是否含有 某个 backend 的特征

    :param module: module 对象
    :return:
    """
    try:
        file = module.__file__
        for backend in SUPPORT_BACKENDS:
            if f'{os.sep}site-packages{os.sep}{backend}' in file:
                return backend
    except:
        pass
    return None


def _get_backend():
    if os.environ.get(FASTNLP_BACKEND, None) != None:
        backend = os.environ.get(FASTNLP_BACKEND)
        logger.debug(f"Get Dataloader backend:{backend} from os.environ")
    else:
        available_backends = set()
        for module in sys.modules.values():
            _backend = _check_module(module)
            if _backend:
                available_backends.add(_backend)
        if len(available_backends) == 1:
            backend = available_backends.pop()
            logger.debug(f"Get Dataloader backend:{backend} from sys.modules.")
        else:
            raise RuntimeError("Fail to detect dataloader backend automatically, please set it manually.")
    return backend
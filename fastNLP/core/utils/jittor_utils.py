__all__ = [
    'is_jittor_module',
    'is_jittor_dataset',
    'jittor_collate_wraps',
]

from collections.abc import Mapping, Callable
from functools import wraps

from fastNLP.envs.imports import _NEED_IMPORT_JITTOR

if _NEED_IMPORT_JITTOR:
    import jittor as jt

from fastNLP.core.dataset import Instance

def is_jittor_module(model) -> bool:
    """
    判断传入的 ``model`` 是否是 :class:`jittor.Module` 类型。

    :param model:
    :return: 当前模型是否为 ``jittor`` 的模型
    """
    try:
        return isinstance(model, jt.Module)
    except BaseException:
        return False

def is_jittor_dataset(dataset) -> bool:
    """
    判断传入的 ``dataset`` 是否是 :class:`jittor.dataset.Dataset` 类型。

    :param dataset:
    :return: 当前 ``dataset`` 是否为 ``jittor`` 的数据集类型
    """
    try:
        if isinstance(dataset, jt.dataset.Dataset):
            return True
        else:
            return False
    except BaseException:
        return False


def jittor_collate_wraps(func, auto_collator: Callable):
    """
    对 ``jittor`` 的 ``collate_fn`` 进行 wrap 封装,。如果数据集为 :class:`Mapping` 类型，那么采用 ``auto_collator`` ，
    否则还是采用 ``jittor`` 的 ``collate_batch``。

    :param func:
    :param auto_collator:
    :return:
    """

    @wraps(func)
    def wrapper(batch):
        if isinstance(batch[0], Instance):
            if auto_collator is not None:
                result = auto_collator(batch)
            else:
                raise ValueError(f"auto_collator is None, but batch exist fastnlp instance!")
        elif isinstance(batch[0], Mapping):
            if auto_collator is not None:
                result = auto_collator(batch)
            else:
                result = func(batch)
        else:
            result = func(batch)
        return result

    return wrapper

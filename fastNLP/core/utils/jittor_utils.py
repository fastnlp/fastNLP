__all__ = [
    'is_jittor_dataset',
    'jittor_collate_wraps'
]

from collections.abc import Mapping, Callable
from functools import wraps

from fastNLP.envs.imports import _NEED_IMPORT_JITTOR

if _NEED_IMPORT_JITTOR:
    import jittor as jt

from fastNLP.core.dataset import Instance


def is_jittor_dataset(dataset) -> bool:
    try:
        if isinstance(dataset, jt.dataset.Dataset):
            return True
        else:
            return False
    except BaseException:
        return False


def jittor_collate_wraps(func, auto_collator: Callable):
    """
    对jittor的collate_fn进行wrap封装, 如果数据集为mapping类型，那么采用auto_collator，否则还是采用jittor自带的collate_batch

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

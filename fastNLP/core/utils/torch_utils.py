from abc import ABC
from typing import Any, Union, Optional
from fastNLP.envs.imports import _NEED_IMPORT_TORCH, _TORCH_GREATER_EQUAL_1_8
DEFAULT_TORCH_GROUP = None
if _NEED_IMPORT_TORCH:
    import torch
    if not _TORCH_GREATER_EQUAL_1_8:
        DEFAULT_TORCH_GROUP = torch.distributed.distributed_c10d.group.WORLD

__all__ = [
    'torch_move_data_to_device'
]

from .utils import apply_to_collection


class TorchTransferableDataType(ABC):
    """
    A custom type for data that can be moved to a torch device via `.to(...)`.
    Example::

        >>> isinstance(dict, TorchTransferableDataType)
        False
        >>> isinstance(torch.rand(2, 3), TorchTransferableDataType)
        True
        >>> class CustomObject:
        ...     def __init__(self):
        ...         self.x = torch.rand(2, 2)
        ...     def to(self, device):
        ...         self.x = self.x.to(device)
        ...         return self
        >>> isinstance(CustomObject(), TorchTransferableDataType)
        True
    """

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> Union[bool, Any]:
        if cls is TorchTransferableDataType:
            to = getattr(subclass, "to", None)
            return callable(to)
        return NotImplemented


def torch_move_data_to_device(batch: Any, device: Optional[Union[str, "torch.device"]] = None,
                              non_blocking: Optional[bool] = True) -> Any:
    r"""
    在 ``pytorch`` 中将数据集合 ``batch`` 传输到给定设备。任何定义方法 ``to(device)`` 的对象都将被移动并且集合中的所有其他对象将保持不变；

    :param batch: 需要迁移的数据；
    :param device: 数据应当迁移到的设备；当该参数的值为 ``None`` 时则不执行任何操作；
    :param non_blocking: ``pytorch`` 的数据迁移方法 ``to`` 的参数；
    :return: 迁移到新设备上的数据集合；
    """
    if device is None:
        return batch

    def batch_to(data: Any) -> Any:
        kwargs = dict(non_blocking=non_blocking) if isinstance(data, torch.Tensor) else {}
        data_output = data.to(device, **kwargs)
        if data_output is not None:
            return data_output
        # user wrongly implemented the `TransferableDataType` and forgot to return `self`.
        return data

    dtype = TorchTransferableDataType
    return apply_to_collection(batch, dtype=dtype, function=batch_to)

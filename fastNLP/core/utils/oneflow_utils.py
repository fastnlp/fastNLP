import os
from typing import Any, Union, Optional
from fastNLP.envs.env import FASTNLP_DISTRIBUTED_CHECK
from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW

if _NEED_IMPORT_ONEFLOW:
    import oneflow

__all__ = [
    'get_oneflow_device',
    'oneflow_move_data_to_device',
    'is_oneflow_module',
    'is_in_oneflow_dist',
]

from .utils import apply_to_collection

def get_oneflow_device(device):
    """
    构造一个 :class:`oneflow.device` 实例并返回。

    :param device: 字符串或 gpu 编号
    :return: :class:`oneflow.device`
    """
    if isinstance(device, oneflow.device):
        return device
    if isinstance(device, int):
        return oneflow.device("cuda", device)
    if isinstance(device, str):
        return oneflow.device(device)
    raise RuntimeError(f"Cannot get `oneflow.device` from {device}.")

def oneflow_move_data_to_device(batch: Any, device: Optional[Union[str, "oneflow.device"]] = None) -> Any:
    r"""
    在 **oneflow** 中将数据集合 ``batch`` 传输到给定设备。

    :param batch: 需要迁移的数据
    :param device: 数据应当迁移到的设备；当该参数的值为 ``None`` 时则不执行任何操作
    :return: 迁移到新设备上的数据集合
    """
    if device is None:
        return batch

    def batch_to(data: Any) -> Any:
        data_output = data.to(device)
        if data_output is not None:
            return data_output
        # user wrongly implemented the `TransferableDataType` and forgot to return `self`.
        return data

    return apply_to_collection(batch, dtype=oneflow.Tensor, function=batch_to)

def is_oneflow_module(model) -> bool:
    """
    判断传入的 ``model`` 是否是 :class:`oneflow.nn.Module` 类型。

    :param model:
    :return: 当前模型是否为 ``oneflow`` 的模型
    """
    try:
        return isinstance(model, oneflow.nn.Module)
    except BaseException:
        return False

def is_in_oneflow_dist() -> bool:
    """
    判断是否处于 **oneflow** 分布式的进程下。
    """
    return "GLOG_log_dir" in os.environ
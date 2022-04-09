__all__ = [
    "paddle_to",
    "paddle_move_data_to_device",
    "get_paddle_gpu_str",
    "get_paddle_device_id",
    "is_in_paddle_dist",
    "is_in_fnlp_paddle_dist",
    "is_in_paddle_launch_dist",
]

import os
import re
from typing import Any, Optional, Union

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.envs import FASTNLP_DISTRIBUTED_CHECK

if _NEED_IMPORT_PADDLE:
    import paddle

from .utils import apply_to_collection


def paddle_to(data, device: Union[str, int]):

    if device == "cpu":
        return data.cpu()
    else:
        return data.cuda(get_paddle_device_id(device))

def get_paddle_gpu_str(device: Union[str, int]):
    """
    获得 `gpu:x` 类型的设备名
    """
    if isinstance(device, str):
        return device.replace("cuda", "gpu")
    return f"gpu:{device}"

def get_paddle_device_id(device: Union[str, int]):
    """
    获得 gpu 的设备id，注意不要传入 `cpu` 。
    """
    if isinstance(device, int):
        return device

    device = device.lower()
    if device == "cpu":
        raise ValueError("Cannot get device id from `cpu`.")

    match_res = re.match(r"gpu:\d+", device)
    if not match_res:
        raise ValueError(
            "The device must be a string which is like 'cpu', 'gpu', 'gpu:x'"
        )
    device_id = device.split(':', 1)[1]
    device_id = int(device_id)

    return device_id

def paddle_move_data_to_device(batch: Any, device: Optional[str] = None,
                              data_device: Optional[str] = None) -> Any:
    r"""
    将数据集合传输到给定设备。只有paddle.Tensor对象会被传输到设备中，其余保持不变

    :param batch:
    :param device: `cpu`, `gpu` or `gpu:x`
    :param data_device:
    :return: 相同的集合，但所有包含的张量都驻留在新设备上；
    """
    if device is None:
        if data_device is not None:
            device = data_device
        else:
            return batch

    def batch_to(data: Any) -> Any:
        return paddle_to(data, device)

    return apply_to_collection(batch, dtype=paddle.Tensor, function=batch_to)

def is_in_paddle_dist():
    """
    判断是否处于分布式的进程下，使用 global_rank 和 selected_gpus 判断
    """
    return ('PADDLE_RANK_IN_NODE' in os.environ and 'FLAGS_selected_gpus' in os.environ)

def is_in_fnlp_paddle_dist():
    """
    判断是否处于 FastNLP 拉起的分布式进程中
    """
    return FASTNLP_DISTRIBUTED_CHECK in os.environ

def is_in_paddle_launch_dist():
    """
    判断是否处于 launch 启动的分布式进程中
    """
    return 'PADDLE_RANK_IN_NODE' in os.environ and \
            'FLAGS_selected_gpus' in os.environ and \
            FASTNLP_DISTRIBUTED_CHECK not in os.environ
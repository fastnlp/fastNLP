__all__ = [
    "get_device_from_visible",
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
from fastNLP.envs import FASTNLP_DISTRIBUTED_CHECK, FASTNLP_BACKEND_LAUNCH, USER_CUDA_VISIBLE_DEVICES

if _NEED_IMPORT_PADDLE:
    import paddle

from .utils import apply_to_collection

def get_device_from_visible(device: Union[str, int], output_type=int):
    """
    在有 CUDA_VISIBLE_DEVICES 的情况下，获取对应的设备。
    如 CUDA_VISIBLE_DEVICES=2,3 ，device=3 ，则返回1。

    :param device: 未转化的设备名
    :param output_type: 返回值的类型
    :return: 转化后的设备id
    """
    if output_type not in [int, str]:
        raise ValueError("Parameter `output_type` should be one of these types: [int, str]")
    if device == "cpu":
        return device
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
    user_visible_devices = os.getenv(USER_CUDA_VISIBLE_DEVICES)
    if user_visible_devices is None:
        raise RuntimeError("`USER_CUDA_VISIBLE_DEVICES` cannot be None, please check if you have set "
                            "`FASTNLP_BACKEND` to 'paddle' before using FastNLP.")
    idx = get_paddle_device_id(device)
    # 利用 USER_CUDA_VISIBLDE_DEVICES 获取用户期望的设备
    if user_visible_devices is None:
        raise RuntimeError("This situation cannot happen, please report a bug to us.")
    idx = user_visible_devices.split(",")[idx]

    cuda_visible_devices_list = cuda_visible_devices.split(',')
    if idx not in cuda_visible_devices_list:
        raise ValueError(f"Can't find your devices {idx} in CUDA_VISIBLE_DEVICES[{cuda_visible_devices}]. ")
    res = cuda_visible_devices_list.index(idx)
    if output_type == int:
        return res
    else:
        return f"gpu:{res}"

def paddle_to(data, device: Union[str, int]):
    """
    将 `data` 迁移到指定的 `device` 上

    :param data: 要迁移的张量
    :param device: 目标设备，可以是 `str` 或 `int`
    :return: 迁移后的张量
    """

    if device == "cpu":
        return data.cpu()
    else:
        # device = get_device_from_visible(device, output_type=int)
        return data.cuda(get_paddle_device_id(device))


def get_paddle_gpu_str(device: Union[str, int]):
    """
    获得 `gpu:x` 类型的设备名

    :param device: 设备编号或设备名
    :return: 返回对应的 `gpu:x` 格式的设备名
    """
    if isinstance(device, str):
        return device.replace("cuda", "gpu")
    return f"gpu:{device}"


def get_paddle_device_id(device: Union[str, int]):
    """
    获得 gpu 的设备id

    :param: device: 设备编号或设备名
    :return: 设备对应的编号
    """
    if isinstance(device, int):
        return device

    device = device.lower()
    if device == "cpu":
        raise ValueError("Cannot get device id from `cpu`.")
    elif device == "gpu":
        return 0

    match_res = re.match(r"gpu:\d+", device)
    if not match_res:
        raise ValueError(
            "The device must be a string which is like 'cpu', 'gpu', 'gpu:x', "
            f"not '{device}'"
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
    return FASTNLP_BACKEND_LAUNCH in os.environ
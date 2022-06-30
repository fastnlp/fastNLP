__all__ = [
    "paddle_to",
    "paddle_move_data_to_device",
    "get_paddle_gpu_str",
    "get_paddle_device_id",
    "is_in_paddle_dist",
    "is_in_fnlp_paddle_dist",
    "is_in_paddle_launch_dist",
    "is_paddle_module",
]

import os
import re
from typing import Any, Optional, Union

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.envs import FASTNLP_DISTRIBUTED_CHECK, FASTNLP_BACKEND_LAUNCH, USER_CUDA_VISIBLE_DEVICES

if _NEED_IMPORT_PADDLE:
    import paddle

from .utils import apply_to_collection


def _convert_data_device(device: Union[str, int]) -> str:
    """
    用于转换 ``driver`` 的 ``data_device`` 的函数。如果用户设置了 ``FASTNLP_BACKEND=paddle``，那么 **fastNLP** 会将
    可见的设备保存在 ``USER_CUDA_VISIBLE_DEVICES`` 中，并且将 ``CUDA_VISIBLE_DEVICES`` 设置为可见的第一张显卡；这是为
    了顺利执行 **paddle** 的分布式训练而设置的。
    
    在这种情况下，单纯使用 ``driver.data_device`` 是无效的。比如在分布式训练中将设备设置为 ``[0,2,3]`` ，且用户设置了
    ``CUDA_VISIBLE_DEVICES=3,4,5,6`` ，那么在 ``rank1``的进程中有::

        os.environ["CUDA_VISIBLE_DEVICES"] = "5"
        os.environ["USER_CUDA_VISIBLE_DEVICES"] = "3,4,5,6"
        driver.data_device = "gpu:2" # 为了向用户正确地反映他们设置的设备减少歧义，因此这里没有设置为 "gpu:5"

    此时我们便需要通过这个函数将 ``data_device`` 转换为 ``gpu:0``。具体过程便是通过索引 **2** 在 ``USER_CUDA_VISIBLE_DEVICES`` 中
    找到设备 **5**，然后在 ``CUDA_VISIBLE_DEVICES`` 中找到设备 **5** 的索引 **0** 返回。

    .. note::

        在分布式单进程仅支持单卡的情况下中，这个函数实际等同于直接转换为 ``gpu:0`` 返回。

    :param device: 未转化的设备；
    :return: 转化后的设备，格式为 ``gpu:x``；
    """
    try:
        user_visible_devices = os.getenv(USER_CUDA_VISIBLE_DEVICES)
        if device == "cpu" or user_visible_devices is None:
            # 传入的是 CPU，或者没有设置 USER_CUDA_VISIBLE_DEVICES
            # 此时不需要进行转换
            return get_paddle_gpu_str(device)

        idx = get_paddle_device_id(device)
        idx = user_visible_devices.split(",")[idx]
        # 此时 CUDA_VISIBLE_DEVICES 一定不是 None
        cuda_visible_devices_list = os.getenv("CUDA_VISIBLE_DEVICES").split(',')
        return f"gpu:{cuda_visible_devices_list.index(idx)}"
    except Exception as e:
        raise ValueError(f"Can't convert device {device} when USER_CUDA_VISIBLE_DEVICES={user_visible_devices} "
                        "and CUDA_VISIBLE_DEVICES={cuda_visible_devices}. If this situation happens, please report this bug to us.")


def paddle_to(data: "paddle.Tensor", device: Union[str, int, 'paddle.fluid.core_avx.Place',
                                                   'paddle.CPUPlace', 'paddle.CUDAPlace']) -> "paddle.Tensor":
    """
    将 ``data`` 迁移到指定的 ``device`` 上。``paddle.Tensor`` 没有类似 ``torch.Tensor`` 的 ``to`` 函数，
    该函数只是集成了 :func:`paddle.Tensor.cpu` 和 :func:`paddle.Tensor.cuda` 两个函数。

    :param data: 要迁移的张量；
    :param device: 目标设备，可以是 ``str`` 或 ``int`` 及 **paddle** 自己的 :class:`paddle.fluid.core_avx.Place`、
        :class:`paddle.CPUPlace` 和 :class:`paddle.CUDAPlace` 类型；
    :return: 迁移后的张量；
    """
    if isinstance(device, paddle.fluid.core_avx.Place):
        if device.is_cpu_place():
            return data.cpu()
        else:
            return data.cuda(device.gpu_device_id())
    elif isinstance(device, paddle.CPUPlace):
        return data.cpu()
    elif isinstance(device, paddle.CUDAPlace):
        return data.gpu(device.get_device_id())
    elif device == "cpu":
        return data.cpu()
    else:
        return data.cuda(get_paddle_device_id(device))


def get_paddle_gpu_str(device: Union[str, int]) -> str:
    """
    获得 ``gpu:x`` 格式的设备名::

        >>> get_paddle_gpu_str(1)
        'gpu:1'
        >>> get_paddle_gpu_str("cuda:1")
        'gpu:1'

    :param device: 设备编号或设备名；
    :return: 返回对应的 ``gpu:x`` 格式的设备名；
    """
    if isinstance(device, str):
        return device.replace("cuda", "gpu")
    return f"gpu:{device}"


def get_paddle_device_id(device: Union[str, int]) -> int:
    """
    获得 ``device`` 的设备编号::

        >>> get_paddle_device_id("gpu:1")
        1
        >>> get_paddle_device_id("gpu")
        0

    请注意不要向这个函数中传入 ``cpu``。

    :param: device: 设备编号或设备名；
    :return: 设备对应的编号；
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

def paddle_move_data_to_device(batch: Any, device: Optional[Union[str, int]]) -> Any:
    r"""
    将 **paddle** 的数据集合传输到给定设备。只有 :class:`paddle.Tensor` 对象会被传输到设备中，其余保持不变。

    :param batch: 需要进行迁移的数据集合；
    :param device: 目标设备。可以是显卡设备的编号，或是``cpu``, ``gpu`` 或 ``gpu:x`` 格式的字符串；
        当这个参数为 `None`` 时，不会执行任何操作。
    :return: 迁移到新设备上的数据集合；
    """
    if device is None:
        return batch

    def batch_to(data: Any) -> Any:
        return paddle_to(data, device)

    return apply_to_collection(batch, dtype=paddle.Tensor, function=batch_to)


def is_in_paddle_dist() -> bool:
    """
    判断是否处于 **paddle** 分布式的进程下，使用 ``PADDLE_RANK_IN_NODE`` 和 ``FLAGS_selected_gpus`` 判断。
    """
    return ('PADDLE_RANK_IN_NODE' in os.environ and 'FLAGS_selected_gpus' in os.environ)


def is_in_fnlp_paddle_dist() -> bool:
    """
    判断是否处于 **fastNLP** 拉起的 **paddle** 分布式进程中
    """
    return FASTNLP_DISTRIBUTED_CHECK in os.environ


def is_in_paddle_launch_dist() -> bool:
    """
    判断是否处于 ``python -m paddle.distributed.launch`` 方法启动的 **paddle** 分布式进程中
    """
    return FASTNLP_BACKEND_LAUNCH in os.environ

def is_paddle_module(model) -> bool:
    """
    判断传入的 ``model`` 是否是 :class:`paddle.nn.Layer` 类型

    :param model: 模型；
    :return: 当前模型是否为 ``paddle`` 的模型；
    """
    try:
        return isinstance(model, paddle.nn.Layer)
    except BaseException:
        return False
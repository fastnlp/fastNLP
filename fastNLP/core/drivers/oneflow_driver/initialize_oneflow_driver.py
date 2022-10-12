import os
from typing import Optional, Union, List, Sequence
from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW
if _NEED_IMPORT_ONEFLOW:
    import oneflow

from .oneflow_driver import OneflowDriver
from .single_device import OneflowSingleDriver
from .ddp import OneflowDDPDriver
from fastNLP.core.log import logger
from fastNLP.envs import FASTNLP_BACKEND_LAUNCH

__all__ = []


def initialize_oneflow_driver(driver: str, device: Optional[Union[str, "oneflow.device", int, List[int]]],
                            model: "oneflow.nn.Module", **kwargs) -> OneflowDriver:
    r"""
    用来根据参数 ``driver` 和 ``device`` 来确定并且初始化一个具体的 ``Driver`` 实例然后返回回去；

    :param driver: 该参数的值应为以下之一：``["oneflow"]``
    :param device: 该参数的格式与 ``Trainer`` 对参数 ``device`` 的要求一致
    :param model: 训练或者评测的具体的模型；

    :return: 一个 :class:`~fastNLP.core.OneflowSingleDriver` 或 :class:`~fastNLP.core.OneflowDDPDriver` 实例；
    """
    # world_size 和 rank
    if FASTNLP_BACKEND_LAUNCH in os.environ:
        if device is not None:
            logger.rank_zero_warning("Parameter `device` would be ignored when you are using `oneflow.distributed.launch` to pull "
                                    "up your script. ", once=True)
        return OneflowDDPDriver(model, None, **kwargs)

    if driver not in {"oneflow"}:
        raise ValueError("Parameter `driver` can only be one of these values: ['oneflow'].")

    _could_use_device_num = oneflow.cuda.device_count()
    if isinstance(device, str):
        device = oneflow.device(device)
    elif isinstance(device, int):
        if device < 0:
            if device != -1:
                raise ValueError("Parameter `device` can only be '-1' when it is smaller than 0.")
            device = [oneflow.device(f"cuda:{w}") for w in range(_could_use_device_num)]
        elif device >= _could_use_device_num:
            raise ValueError("The gpu device that parameter `device` specifies is not existed.")
        else:
            device = oneflow.device(f"cuda:{device}")
    elif isinstance(device, Sequence):
        device = list(set(device))
        for each in device:
            if not isinstance(each, int):
                raise ValueError("When parameter `device` is 'Sequence' type, the value in it should be 'int' type.")
            elif each < 0:
                raise ValueError("When parameter `device` is 'Sequence' type, the value in it should be bigger than 0.")
            elif each >= _could_use_device_num:
                raise ValueError(f"When parameter `device` is 'Sequence' type, the value in it should not be bigger than"
                                 f" the available gpu number:{_could_use_device_num}.")
        device = [oneflow.device(f"cuda:{w}") for w in device]
    elif device is not None and not isinstance(device, oneflow.device):
        raise ValueError("Parameter `device` is wrong type, please check our documentation for the right use.")

    if driver == "oneflow":  # single, ddp, 直接启动。
        if not isinstance(device, List):
            return OneflowSingleDriver(model, device, **kwargs)
        else:
            raise RuntimeError("If you want to run distributed training, please use "
                                "'python -m oneflow.distributed.launch xxx.py'.")
            return OneflowDDPDriver(model, device, **kwargs)
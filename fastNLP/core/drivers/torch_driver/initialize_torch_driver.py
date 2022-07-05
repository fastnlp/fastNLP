import os
from typing import Optional, Union, List, Sequence
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch

from .torch_driver import TorchDriver
from .single_device import TorchSingleDriver
from .ddp import TorchDDPDriver
from .fairscale import FairScaleDriver
from .deepspeed import DeepSpeedDriver
from .torch_fsdp import TorchFSDPDriver
from fastNLP.core.log import logger
from fastNLP.envs import FASTNLP_BACKEND_LAUNCH
from pkg_resources import parse_version

__all__ = []


def initialize_torch_driver(driver: str, device: Optional[Union[str, "torch.device", int, List[int]]],
                            model: "torch.nn.Module", **kwargs) -> TorchDriver:
    r"""
    用来根据参数 ``driver` 和 ``device`` 来确定并且初始化一个具体的 ``Driver`` 实例然后返回回去；

    :param driver: 该参数的值应为以下之一：``["torch", "fairscale", "deepspeed"]``；
    :param device: 该参数的格式与 ``Trainer`` 对参数 ``device`` 的要求一致；
    :param model: 训练或者评测的具体的模型；

    :return: 返回一个 :class:`~fastNLP.core.TorchSingleDriver` 或 :class:`~fastNLP.core.TorchDDPDriver` 实例；
    """
    if parse_version(torch.__version__) < parse_version('1.6'):
        raise RuntimeError(f"Pytorch(current version:{torch.__version__}) need to be older than 1.6.")
    # world_size 和 rank
    if FASTNLP_BACKEND_LAUNCH in os.environ:
        if device is not None:
            logger.rank_zero_warning("Parameter `device` would be ignored when you are using `torch.distributed.run` to pull "
                           "up your script. And we will directly get the local device via "
                           "`os.environ['LOCAL_RANK']`.", once=True)
        if driver == 'fairscale':
            return FairScaleDriver(model, torch.device(f"cuda:{os.environ['LOCAL_RANK']}"),
                                   is_pull_by_torch_run=True, **kwargs)
        elif driver == 'deepspeed':
            return DeepSpeedDriver(model, torch.device(f"cuda:{os.environ['LOCAL_RANK']}"),
                                   is_pull_by_torch_run=True, **kwargs)
        else:
            return TorchDDPDriver(model, torch.device(f"cuda:{os.environ['LOCAL_RANK']}"),
                                  is_pull_by_torch_run=True, **kwargs)

    if driver not in {"torch", "fairscale", "deepspeed", "torch_fsdp"}:
        raise ValueError("Parameter `driver` can only be one of these values: ['torch', 'fairscale'].")

    _could_use_device_num = torch.cuda.device_count()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        if device < 0:
            if device != -1:
                raise ValueError("Parameter `device` can only be '-1' when it is smaller than 0.")
            device = [torch.device(f"cuda:{w}") for w in range(_could_use_device_num)]
        elif device >= _could_use_device_num:
            print(device, _could_use_device_num)
            raise ValueError("The gpu device that parameter `device` specifies is not existed.")
        else:
            device = torch.device(f"cuda:{device}")
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
        device = [torch.device(f"cuda:{w}") for w in device]
    elif device is not None and not isinstance(device, torch.device):
        raise ValueError("Parameter `device` is wrong type, please check our documentation for the right use.")

    if driver == "torch":  # single, ddp, 直接启动。
        if not isinstance(device, List):
            return TorchSingleDriver(model, device, **kwargs)
        else:
            return TorchDDPDriver(model, device, **kwargs)
    elif driver == "fairscale":
        if not isinstance(device, List):
            if device.type == 'cpu':
                raise ValueError("You are using `fairscale` driver, but your chosen `device` is 'cpu'.")
            logger.warning_once("Notice you are using `fairscale`, but the `device` is only one gpu.")
            return FairScaleDriver(model, [device], **kwargs)
        else:
            return FairScaleDriver(model, device, **kwargs)
    elif driver == "deepspeed":
        if not isinstance(device, List):
            if device.type == 'cpu':
                raise ValueError("You are using `deepspeed` driver, but your chosen `device` is 'cpu'.")
            logger.warning_once("Notice you are using `deepspeed`, but the `device` is only one gpu.")
            return DeepSpeedDriver(model, [device], **kwargs)
        else:
            return DeepSpeedDriver(model, device, **kwargs)
    elif driver == "torch_fsdp":
        if not isinstance(device, List):
            if device.type == 'cpu':
                raise ValueError("You are using `torch_fsdp` driver, but your chosen `device` is 'cpu'.")
            logger.warning_once("Notice you are using `torch_fsdp`, but the `device` is only one gpu.")
            return TorchFSDPDriver(model, [device], **kwargs)
        else:
            return TorchFSDPDriver(model, device, **kwargs)
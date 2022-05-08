import os
from typing import Optional, Union, List, Sequence
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch

from .torch_driver import TorchDriver
from .single_device import TorchSingleDriver
from .ddp import TorchDDPDriver
from fastNLP.core.log import logger
from fastNLP.envs import FASTNLP_BACKEND_LAUNCH


def initialize_torch_driver(driver: str, device: Optional[Union[str, "torch.device", int, List[int]]],
                            model: "torch.nn.Module", **kwargs) -> TorchDriver:
    r"""
    用来根据参数 `driver` 和 `device` 来确定并且初始化一个具体的 `Driver` 实例然后返回回去；
    注意如果输入的 `device` 如果和 `driver` 对应不上就直接报错；

    :param driver: 该参数的值应为以下之一：["torch", "torch_ddp", "fairscale"]；
    :param device: 该参数的格式与 `Trainer` 对参数 `device` 的要求一致；
    :param model: 训练或者评测的具体的模型；

    :return: 返回一个元组，元组的第一个值是具体的基于 pytorch 的 `Driver` 实例，元组的第二个值是该 driver 的名字（用于检测一个脚本中
     先后 driver 的次序的正确问题）；
    """
    # world_size 和 rank
    if FASTNLP_BACKEND_LAUNCH in os.environ:
        if device is not None:
            logger.warning_once("Parameter `device` would be ignored when you are using `torch.distributed.run` to pull "
                           "up your script. And we will directly get the local device via "
                           "`os.environ['LOCAL_RANK']`.")
        return TorchDDPDriver(model, torch.device(f"cuda:{os.environ['LOCAL_RANK']}"), True, **kwargs)

    if driver not in {"torch", "torch_ddp", "fairscale"}:
        raise ValueError("Parameter `driver` can only be one of these values: ['torch', 'torch_ddp', 'fairscale'].")

    _could_use_device_num = torch.cuda.device_count()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        if device < 0:
            if device != -1:
                raise ValueError("Parameter `device` can only be '-1' when it is smaller than 0.")
            device = [torch.device(f"cuda:{w}") for w in range(_could_use_device_num)]
        elif device >= _could_use_device_num:
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
                raise ValueError("When parameter `device` is 'Sequence' type, the value in it should not be bigger than"
                                 " the available gpu number.")
        device = [torch.device(f"cuda:{w}") for w in device]
    elif device is not None and not isinstance(device, torch.device):
        raise ValueError("Parameter `device` is wrong type, please check our documentation for the right use.")

    if driver == "torch":  # single, ddp, 直接启动。
        if not isinstance(device, List):
            return TorchSingleDriver(model, device, **kwargs)
        else:
            logger.info("Notice you are using `torch` driver but your chosen `device` are multi gpus, we will use "
                           "`TorchDDPDriver` by default. But if you mean using `TorchDDPDriver`, you should choose parameter"
                           "`driver` as `TorchDDPDriver`.")
            return TorchDDPDriver(model, device, **kwargs)
    elif driver == "torch_ddp":
        if device is not None and not isinstance(device, List):
            if device.type == 'cpu':
                raise ValueError("You are using `torch_ddp` driver, but your chosen `device` is 'cpu'.")
            logger.info("Notice you are using `torch_ddp` driver, but your chosen `device` is only one gpu, we will "
                        "still use `TorchDDPDriver` for you, but if you mean using `torch_ddp`, you should "
                        "choose `torch` driver.")
            return TorchDDPDriver(model, [device], **kwargs)
        else:
            return TorchDDPDriver(model, device, **kwargs)
    elif driver == "fairscale":
        raise NotImplementedError("`fairscale` is not support right now.")
        # if not isinstance(device, List):
        #     if device.type == 'cpu':
        #         raise ValueError("You are using `fairscale` driver, but your chosen `device` is 'cpu'.")
        #     log.info("Notice you are using `fairscale` driver, but your chosen `device` is only one gpu, we will"
        #                 "still use `fairscale` for you, but if you mean using `TorchSingleDriver`, you should "
        #                 "choose `torch` driver.")
        #     return ShardedDriver(model, [device], **kwargs)
        # else:
        #     return ShardedDriver(model, device, **kwargs)
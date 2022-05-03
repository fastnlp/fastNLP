import os

from typing import Optional, List, Sequence, Union

from .paddle_driver import PaddleDriver
from .single_device import PaddleSingleDriver
from .fleet import PaddleFleetDriver

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.core.utils import is_in_paddle_launch_dist
from fastNLP.core.log import logger

if _NEED_IMPORT_PADDLE:
    import paddle

def initialize_paddle_driver(driver: str, device: Optional[Union[str, int, List[int]]],
                            model: "paddle.nn.Layer", **kwargs) -> PaddleDriver:
    r"""
    用来根据参数 `driver` 和 `device` 来确定并且初始化一个具体的 `Driver` 实例然后返回回去；
    1、如果检测到当前进程为用户通过 `python -m paddle.distributed.launch xxx.py` 方式拉起的，则将
    设备自动设置为用户指定的设备（由于我们在引入 fastNLP 进行了特殊的设置，因此可以通过 `CUDA_VISIBLE_DEVICES` 获取）
    2、如果检测到输入的 `driver` 是 `paddle` 但 `device` 包含了多个设备，那么我们会给出警告并且自动返回多卡的 Driver
    3、如果检测到输入的 `driver` 是 `fleet` 但 `device` 仅有一个设备，那么我们会给出警告但仍旧返回多卡的 Driver

    :param driver: 该参数的值应为以下之一：["paddle", "fleet"]；
    :param device: 该参数的格式与 `Trainer` 对参数 `device` 的要求一致；
    :param model: 训练或者评测的具体的模型；

    :return: 返回构造的 `Driver` 实例。
    """
    if is_in_paddle_launch_dist():
        if device is not None:
            logger.warning_once("Parameter `device` would be ignored when you are using `paddle.distributed.launch` to pull "
                           "up your script. And we will directly get the local device via "
                           "and `os.environ['CUDA_VISIBLE_DEVICES']``.")
        device = [int(g) for g in os.environ["CUDA_VISIBLE_DEVICES"].split(",")]
        # TODO 目前一个进程仅对应一个卡，所以暂时传入一个 int
        return PaddleFleetDriver(model, device[0], True, **kwargs)

    if driver not in {"paddle", "fleet"}:
        raise ValueError("Parameter `driver` can only be one of these values: ['paddle', 'fleet'].")

    user_visible_devices = os.getenv("USER_CUDA_VISIBLE_DEVICES")
    if user_visible_devices is None:
        raise RuntimeError("`USER_CUDA_VISIBLE_DEVICES` cannot be None, please check if you have set "
                            "`FASTNLP_BACKEND` to 'paddle' before using FastNLP.")
    _could_use_device_num = len(user_visible_devices.split(","))
    if isinstance(device, int):
        if device < 0 and device != -1:
            raise ValueError("Parameter `device` can only be '-1' when it is smaller than 0.")
        if device >= _could_use_device_num:
            raise ValueError("The gpu device that parameter `device` specifies is not existed.")
        if device == -1:
            device = list(range(_could_use_device_num))
    elif isinstance(device, Sequence) and not isinstance(device, str):
        device = list(set(device))
        for each in device:
            if not isinstance(each, int):
                raise ValueError("When parameter `device` is 'Sequence' type, the value in it should be 'int' type.")
            elif each < 0:
                raise ValueError("When parameter `device` is 'Sequence' type, the value in it should be bigger than 0.")
            elif each >= _could_use_device_num:
                raise ValueError("When parameter `device` is 'Sequence' type, the value in it should not be bigger than"
                                 " the available gpu number.")
    elif device is not None and not isinstance(device, str):
        raise ValueError("Parameter `device` is wrong type, please check our documentation for the right use.")

    if driver == "paddle":
        if not isinstance(device, List):
            return PaddleSingleDriver(model, device, **kwargs)
        else:
            logger.warning("Notice you are using `paddle` driver but your chosen `device` are multi gpus, we will use"
                            "`Fleetriver` by default. But if you mean using `PaddleFleetDriver`, you should choose parameter"
                            "`driver` as `PaddleFleetDriver`.")
            return PaddleFleetDriver(model, device, **kwargs)
    elif driver == "fleet":
        if not isinstance(device, List):
            if device == "cpu":
                raise ValueError("You are using `fleet` driver, but your chosen `device` is 'cpu'.")
            logger.warning("Notice you are using `fleet` driver, but your chosen `device` is only one gpu, we will"
                            "still use `PaddleFleetDriver` for you, but if you mean using `PaddleSingleDriver`, you should "
                            "choose `paddle` driver.")
            return PaddleFleetDriver(model, [device], **kwargs)
        else:
            return PaddleFleetDriver(model, device, **kwargs)

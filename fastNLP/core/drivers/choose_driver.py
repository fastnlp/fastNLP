from typing import Union, Optional, List

from .driver import Driver
from ..utils import is_torch_module, is_paddle_module, is_jittor_module, is_oneflow_module

__all__ = []

def choose_driver(model, driver: Union[str, Driver], device: Optional[Union[int, List[int], str]], **kwargs) -> Driver:
    r"""
    根据输入的参数 ``driver`` 和 ``device`` 的格式来决定具体的工作模式。

    :param model: 运行过程中使用的具体的最原始的模型。
    :param driver: 训练模型所使用的具体的驱动模式，应当为以下选择中的一个：``["auto", "torch", "paddle", "jittor", "fairscale", "deepspeed", "oneflow", "torch_fsdp"]``，分别对应
        各种框架。值为 ``'auto'`` 时，将会根据模型的类型进行选择。
    :param device: 训练使用的设备。详细的格式可以查阅 :class:`~fastNLP.core.controllers.Trainer` 中的说明。
    :param kwargs: 其余的传给 `Driver` 的参数。
    """

    # 如果用户直接传进来一个 driver 实例，我们就直接返回回去，目前用户需要自己保证传进来的 driver 的正确性；
    if isinstance(driver, Driver):
        return driver

    if driver == "auto":
        if is_torch_module(model):
            driver = "torch"
        elif is_paddle_module(model):
            driver = "paddle"
        elif is_jittor_module(model):
            driver = "jittor"
        elif is_oneflow_module(model):
            driver = "oneflow"
        else:
            raise ValueError(f"Cannot choose driver automatically based on model, please set `driver` specifically.")

    if driver in {"torch", "fairscale", "deepspeed", "torch_fsdp"}:
        from fastNLP.core.drivers.torch_driver.initialize_torch_driver import initialize_torch_driver
        return initialize_torch_driver(driver, device, model, **kwargs)
    elif driver in {"jittor"}:
        from fastNLP.core.drivers.jittor_driver.initialize_jittor_driver import initialize_jittor_driver
        return initialize_jittor_driver(driver, device, model, **kwargs)
    elif driver in {"paddle"}:
        from fastNLP.core.drivers.paddle_driver.initialize_paddle_driver import initialize_paddle_driver
        return initialize_paddle_driver(driver, device, model, **kwargs)
    elif driver in {"oneflow"}:
        from fastNLP.core.drivers.oneflow_driver.initialize_oneflow_driver import initialize_oneflow_driver
        return initialize_oneflow_driver(driver, device, model, **kwargs)
    else:
        raise ValueError("Parameter `driver` can only be one of these values: ['torch', 'fairscale', "
                         "'jittor', 'paddle', 'oneflow'].")
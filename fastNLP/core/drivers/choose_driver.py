from typing import Union, Optional, List

from .driver import Driver


def choose_driver(model, driver: Union[str, Driver], device: Optional[Union[int, List[int], str]], **kwargs) -> Driver:
    r"""
    根据输入的参数 'gpus' 的格式来决定具体的工作模式;

    :param model: 运行过程中使用的具体的最原始的模型；
    :param driver: 应当为字符串或者 `Driver` 实例，表示运行中具体使用的训练/评测模式；
    :param device: 具体的形式请参见 `fastNLP.core.drivers.torch_driver.utils.initialize_torch_dirver` 的注释；
    :param kwargs: 其余的传给 `Driver` 的参数；
    """

    # 如果用户直接传进来一个 driver 实例，我们就直接返回回去，目前用户需要自己保证传进来的 driver 的正确性；
    if isinstance(driver, Driver):
        return driver

    if driver in {"torch", "fairscale", "deepspeed"}:
        from fastNLP.core.drivers.torch_driver.initialize_torch_driver import initialize_torch_driver
        return initialize_torch_driver(driver, device, model, **kwargs)
    elif driver in {"jittor"}:
        from fastNLP.core.drivers.jittor_driver.initialize_jittor_driver import initialize_jittor_driver
        return initialize_jittor_driver(driver, device, model, **kwargs)
    elif driver in {"paddle"}:
        from fastNLP.core.drivers.paddle_driver.initialize_paddle_driver import initialize_paddle_driver
        return initialize_paddle_driver(driver, device, model, **kwargs)
    else:
        raise ValueError("Parameter `driver` can only be one of these values: ['torch', 'fairscale', "
                         "'jittor', 'paddle'].")
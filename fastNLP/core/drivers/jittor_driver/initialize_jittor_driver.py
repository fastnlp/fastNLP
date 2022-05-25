from typing import Union, List

from fastNLP.core.drivers.jittor_driver.mpi import JittorMPIDriver
from fastNLP.core.drivers.jittor_driver.jittor_driver import JittorDriver
from fastNLP.core.drivers.jittor_driver.single_device import JittorSingleDriver
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR

if _NEED_IMPORT_JITTOR:
    import jittor

__all__ = []

def initialize_jittor_driver(driver: str, device: Union[str, int, List[int]], model: jittor.Module, **kwargs) -> JittorDriver:
    r"""
    用来根据参数 ``device`` 来确定并且初始化一个具体的 ``Driver`` 实例然后返回回去。

    .. todo::

        创建多卡的 driver

    :param driver: 该参数的值应为以下之一：``["jittor"]``；
    :param device: ``jittor`` 运行的设备；
    :param model: 训练或者评测的具体的模型；
    :param kwargs: 

    :return: :class:`~fastNLP.core.JittorSingleDriver` 或 :class:`~fastNLP.core.JittorMPIDriver` 实例；
    """

    if driver not in {"jittor"}:
        raise ValueError("Parameter `driver` can only be one of these values: ['jittor'].")

    # TODO 实现更详细的判断
    if device in ["cpu", "gpu", "cuda", "cuda:0", 0, None]:
        return JittorSingleDriver(model, device, **kwargs)
    elif type(device) is int:
        return JittorMPIDriver(model, device, **kwargs)
    elif type(device) is list:
        return JittorMPIDriver(model, device, **kwargs)
    else:
        raise NotImplementedError(f"Device={device}")

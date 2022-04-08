from typing import Union, List

from fastNLP.core.drivers.jittor_driver.jittor_driver import JittorDriver
from fastNLP.core.drivers.jittor_driver.single_device import JittorSingleDriver
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR

if _NEED_IMPORT_JITTOR:
    import jittor

def initialize_jittor_driver(driver: str, device: Union[str, int, List[int]], model: jittor.Module, **kwargs) -> JittorDriver:
    r"""
    用来根据参数 `driver` 和 `device` 来确定并且初始化一个具体的 `Driver` 实例然后返回回去；
    在这个函数中，我们会根据用户设置的device来确定JittorDriver的mode。

    :param driver: 该参数的值应为以下之一：["jittor"]；
    :param device: jittor运行的设备
    :param model: 训练或者评测的具体的模型；
    :param kwargs: 

    :return: 返回一个元组，元组的第一个值是具体的基于 jittor 的 `Driver` 实例，元组的第二个值是该 driver 的名字（用于检测一个脚本中
     先后 driver 的次序的正确问题）；
    """

    if driver not in {"jittor"}:
        raise ValueError("Parameter `driver` can only be one of these values: ['jittor'].")

    # TODO 实现更详细的判断
    if driver == "jittor":
        return JittorSingleDriver(model, device, **kwargs)
    else:
        raise NotImplementedError
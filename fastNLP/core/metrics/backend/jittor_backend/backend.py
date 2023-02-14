from typing import List, Union

import numpy as np

from fastNLP.core.metrics.backend import Backend
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR

if _NEED_IMPORT_JITTOR:
    import jittor

__all__ = []  # type: ignore


class JittorBackend(Backend):

    def __init__(self):
        super(JittorBackend, self).__init__()
        self._specified = True

    def aggregate(self, tensor, method: str):
        """聚集结果，并根据 method 计算后，返回结果。"""
        return tensor

    def create_tensor(self, value: Union[float, List]):
        """创建 tensor，并且填入 value 作为值。"""
        tensor = jittor.array(value)
        return tensor

    def fill_value(self, tensor, value: Union[float, List, np.ndarray]):
        """将 tensor 的值设置为 value。"""
        length = len(value) if isinstance(value, (np.ndarray, list)) else 1
        value = jittor.array(value)
        for i in range(length):
            tensor[i] = value[i]
        return tensor

    def get_scalar(self, tensor) -> float:
        """tensor 的 scalar 值。

        :param tensor:
        :return:
        """
        return tensor.item()

    def to_list(self, tensor) -> List:
        """``tensor`` 的 value 值。

        :param tensor: 传入的张量;
        :return:
        """
        return tensor.tolist()

    def is_specified(self) -> bool:
        """判断是否是某种框架的 backend.

        :return:
        """
        return self._specified

    def tensor2numpy(self, tensor):
        """将 tensor 转为 numpy.

        :param tensor:
        :return:
        """
        if isinstance(tensor, jittor.Var):
            return tensor.detach().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            raise ValueError(f'tensor: {tensor} can not convert to ndarray!')

    def move_tensor_to_device(self, tensor, device):
        """jittor 的没有转移设备的函数，因此该函数实际上无效。"""
        return tensor

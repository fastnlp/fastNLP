import numpy as np

from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
from fastNLP.core.metrics.backend import Backend

if _NEED_IMPORT_JITTOR:
    import jittor


class JittorBackend(Backend):

    def __init__(self):
        super(JittorBackend, self).__init__()
        self._specified = True

    def aggregate(self, tensor, method: str):
        """
        聚集结果，并根据method计算后，返回结果
        """
        return tensor

    def create_tensor(self, value: float):
        """
        创建tensor，并且填入value作为值
        """
        value = jittor.Var(value)
        return value

    def fill_value(self, tensor, value: float):
        """
        将tensor的值设置为value

        """
        value = jittor.full_like(tensor, value)
        return value

    def get_scalar(self, tensor) -> float:
        """
        tensor的saclar值

        :param tensor:
        :return:
        """
        return tensor.item()

    def is_specified(self) -> bool:
        """
        判断是否是某种框架的backend

        :return:
        """
        return self._specified

    def tensor2numpy(self, tensor):
        """
        将tensor转为numpy

        :param tensor:
        :return:
        """
        if isinstance(tensor, jittor.Var):
            return tensor.detach().numpy()
        elif isinstance(tensor, np.array):
            return tensor
        else:
            raise ValueError(f"tensor: {tensor} can not convert to ndarray!")

    def move_tensor_to_device(self, tensor, device):
        """
        jittor的没有转移设备的函数，因此该函数实际上无效
        """
        return tensor

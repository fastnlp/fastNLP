from typing import List

import numpy as np

from fastNLP.core.metrics.backend import Backend
from fastNLP.core.metrics.utils import AggregateMethodError
from fastNLP.core.utils import is_in_oneflow_dist
from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW
from fastNLP.core.drivers.oneflow_driver.dist_utils import fastnlp_oneflow_all_gather


if _NEED_IMPORT_ONEFLOW:
    import oneflow
    import oneflow.comm as comm

__all__ = []

class OneflowBackend(Backend):
    def __init__(self):
        super().__init__()
        self._specified = True

    def aggregate(self, tensor, method: str):
        """
        聚集结果，并根据 method 计算后，返回结果

        :param tensor: 需要聚合的张量
        :param method: 聚合的方法， 目前支持 ``['sum', 'mean', 'max', 'min']``:

            * method 为 ``'sum'`` 时， 会将多张卡上聚合结果在维度为 `0` 上 累加起来。
            * method 为 ``'mean'`` 时，会将多张卡上聚合结果在维度为 `0` 上取平均值。
            * method 为 ``'max'`` 时，会将多张卡上聚合结果在维度为 `0` 上取最大值。
            * method 为 ``'min'`` 时，会将多张卡上聚合结果在维度为 `0` 上取最小值。

        """
        if isinstance(tensor, oneflow.Tensor):
            # TODO 暂时没有找到 oneflow 中检测是否初始化了分布式环境的方法
            if is_in_oneflow_dist():
                if method is None:
                    raise AggregateMethodError(should_have_aggregate_method=True)
                tensor = self.all_gather_object(tensor)
                if isinstance(tensor[0], oneflow.Tensor):
                    tensor = oneflow.stack(tensor)
                # 第一步, aggregate结果
                if method == 'sum':
                    tensor = oneflow.sum(tensor, dim=0)
                elif method == 'mean':
                    tensor = oneflow.mean(tensor, dim=0)
                elif method == 'max':
                    tensor, _ = oneflow.max(tensor, dim=0)
                elif method == 'min':
                    tensor, _ = oneflow.min(tensor, dim=0)
                else:
                    raise AggregateMethodError(should_have_aggregate_method=False)

        return tensor

    def create_tensor(self, value: float):
        """
        创建 tensor，并且填入 value 作为值

        :param value: 创建张量的初始值
        """
        tensor = oneflow.ones(1).fill_(value)
        return tensor

    def fill_value(self, tensor, value: float):
        """
        将 tensor 的值设置为 value

        :param tensor: 传入的张量
        :param value: 需要 fill 的值。
        """
        tensor.fill_(value)
        return tensor

    def get_scalar(self, tensor) -> float:
        """
        获取 tensor 的 scalar 值

        :param tensor: 传入的张量
        """
        return tensor.item()

    def tensor2numpy(self, tensor) -> np.array:
        """
        将 tensor 转为 numpy 值， 主要是在 metric 计算中使用

        :param tensor: 传入的张量
        """

        if isinstance(tensor, oneflow.Tensor):
            return tensor.cpu().detach().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        elif isinstance(tensor, (float, int)):
            return tensor
        else:
            raise ValueError(f"tensor: {tensor} can not convert to ndarray!")

    @staticmethod
    def is_distributed() -> bool:
        """
        判断是否为 ddp 状态

        :return:
        """
        return is_in_oneflow_dist()

    def move_tensor_to_device(self, tensor, device):
        """
        将张量移到设备上

        :param tensor: 需要移动的张量
        :param device: 设备名， 一般为 "cpu", "cuda:0"等字符串
        """
        return tensor.to(device)

    def all_gather_object(self, obj, group=None) -> List:
        """
        给定 obj 将各个 rank 上的 obj 汇总到每个 obj 上。返回一个 list 对象，里面依次为各个 rank 对应的 obj 。

        :param obj:
        :param group:
        """
        if self.is_distributed():
            obj_list = fastnlp_oneflow_all_gather(obj)
            return obj_list
        return [obj]


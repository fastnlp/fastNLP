import os
from typing import List, Any, Union

import numpy as np

from fastNLP.core.metrics.backend import Backend
from fastNLP.core.utils.paddle_utils import paddle_to, _convert_data_device, is_in_paddle_dist
from fastNLP.core.metrics.utils import AggregateMethodError
from fastNLP.core.drivers.paddle_driver.dist_utils import fastnlp_paddle_all_gather
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE

if _NEED_IMPORT_PADDLE:
    import paddle
    import paddle.distributed as dist
    from paddle.fluid.dygraph import parallel_helper

__all__ = []

class PaddleBackend(Backend):
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
        if isinstance(tensor, paddle.Tensor):
            if parallel_helper._is_parallel_ctx_initialized():
                if method is None:
                    raise AggregateMethodError(should_have_aggregate_method=True)
                tensor = self.all_gather_object(tensor)
                if isinstance(tensor[0], paddle.Tensor):
                    tensor = paddle.stack(tensor)
                # 第一步, aggregate结果
                if method == 'sum':
                    tensor = paddle.sum(tensor, axis=0)
                elif method == 'mean':
                    tensor = paddle.mean(tensor, axis=0)
                elif method == 'max':
                    tensor, _ = paddle.max(tensor, axis=0)
                elif method == 'min':
                    tensor, _ = paddle.min(tensor, axis=0)
                else:
                    raise AggregateMethodError(should_have_aggregate_method=False)

        return tensor

    def create_tensor(self, value: Union[float,List]):
        """
        创建 tensor，并且填入 value 作为值

        :param value: 创建张量的初始值
        """
        tensor = paddle.to_tensor(value)
        return tensor

    def fill_value(self, tensor, value: Union[float,List]):
        """
        将 tensor 的值设置为 value

        :param tensor: 传入的张量
        :param value: 需要 fill 的值。
        """
        paddle.assign(paddle.to_tensor(value),tensor)
        return tensor

    def get_scalar(self, tensor) -> float:
        """
        获取 tensor 的 scalar 值

        :param tensor: 传入的张量
        """
        return tensor.item()

    def to_list(self, tensor) -> List:
        """
       ``tensor`` 的 value 值.

       :param tensor: 传入的张量;
       :return:
       """
        return tensor.tolist()


    def tensor2numpy(self, tensor) -> np.array:
        """
        将 tensor 转为 numpy 值， 主要是在 metric 计算中使用

        :param tensor: 传入的张量
        """
        if isinstance(tensor, paddle.Tensor):
            return tensor.cpu().detach().numpy()
        elif isinstance(tensor, np.array):
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
        return is_in_paddle_dist()

    def move_tensor_to_device(self, tensor, device):
        """
        将张量移到设备上

        :param tensor: 需要移动的张量
        :param device: 设备名， 一般为 "cpu", "cuda:0"等字符串
        """
        device = _convert_data_device(device)
        return paddle_to(tensor, device)

    def all_gather_object(self, obj, group=None) -> List:
        """
        给定 obj 将各个 rank 上的 obj 汇总到每个 obj 上。返回一个 list 对象，里面依次为各个 rank 对应的 obj 。

        :param obj:
        :param group:
        """
        if self.is_distributed():
            obj_list = fastnlp_paddle_all_gather(obj, group=group)
            return obj_list
        return [obj]

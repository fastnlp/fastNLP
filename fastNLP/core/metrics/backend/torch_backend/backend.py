from typing import Any, List, Optional

import numpy as np

from fastNLP.core.metrics.backend import Backend
from fastNLP.core.metrics.utils import AggregateMethodError
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from fastNLP.core.drivers.torch_driver.dist_utils import fastnlp_torch_all_gather


if _NEED_IMPORT_TORCH:
    import torch
    import torch.distributed as dist

__all__ = []

class TorchBackend(Backend):
    def __init__(self):
        super().__init__()
        self._specified = True

    def aggregate(self, tensor, method: str):
        """
        聚集结果，并根据method计算后，返回结果。
        """
        if isinstance(tensor, torch.Tensor):
            if dist.is_initialized():
                if method is None:
                    raise AggregateMethodError(should_have_aggregate_method=True)
                tensor = self.all_gather_object(tensor)
                if isinstance(tensor[0], torch.Tensor):
                    tensor = torch.stack(tensor)
                # 第一步, aggregate结果
                if method == 'sum':
                    tensor = torch.sum(tensor, dim=0)
                elif method == 'mean':
                    tensor = torch.mean(tensor, dim=0)
                elif method == 'max':
                    tensor, _ = torch.max(tensor, dim=0)
                elif method == 'min':
                    tensor, _ = torch.min(tensor, dim=0)
                else:
                    raise AggregateMethodError(should_have_aggregate_method=False)

        return tensor

    def create_tensor(self, value: float):
        """
        创建tensor，并且填入value作为值
        """
        tensor = torch.ones(1).fill_(value)
        return tensor

    def fill_value(self, tensor, value: float):
        """
        将tensor的值设置为value

        """
        tensor.fill_(value)
        return tensor

    def get_scalar(self, tensor) -> float:
        return tensor.item()

    def tensor2numpy(self, tensor) -> np.array:
        """
        将对应的tensor转为numpy对象

        """

        if isinstance(tensor, torch.Tensor):
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
        :return:
        """
        return dist.is_available() and dist.is_initialized()

    def move_tensor_to_device(self, tensor, device):
        return tensor.to(device)

    def all_gather_object(self, obj, group=None) -> List:
        if self.is_distributed():
            obj_list = fastnlp_torch_all_gather(obj, group=group)
            return obj_list
        return [obj]


from typing import Any, List, Optional

import numpy as np

from fastNLP.core.metrics.backend import Backend
from fastNLP.core.metrics.utils import AggregateMethodError
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from fastNLP.core.drivers.torch_driver.dist_utils import fastnlp_torch_all_gather


if _NEED_IMPORT_TORCH:
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F


def _simple_gather_all_tensors(result, group: Any, world_size: int) -> List:
    gathered_result = [torch.zeros_like(result) for _ in range(world_size)]
    dist.all_gather(gathered_result, result, group)
    return gathered_result


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
                tensor = self._gather_all(tensor)
                # tensor = self.all_gather_object(tensor)
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

    @staticmethod
    def _gather_all(result, group: Optional[Any] = None) -> List:
        """Function to gather all tensors from several ddp processes onto a list that is broadcasted to all processes.
        Works on tensors that have the same number of dimensions, but where each dimension may differ. In this case
        tensors are padded, gathered and then trimmed to secure equal workload for all processes.

        Args:
            result: the value to sync
            group: the process group to gather results from. Defaults to all processes (world)

        Return:
            gathered_result: list with size equal to the process group where
                gathered_result[i] corresponds to result tensor from process i
        """

        if group is None:
            group = dist.group.WORLD

        # convert tensors to contiguous format
        result = result.contiguous()

        world_size = dist.get_world_size(group)
        dist.barrier(group=group)

        # if the tensor is scalar, things are easy
        if result.ndim == 0:
            return _simple_gather_all_tensors(result, group, world_size)

        # 1. Gather sizes of all tensors
        local_size = torch.tensor(result.shape, device=result.device)
        local_sizes = [torch.zeros_like(local_size) for _ in range(world_size)]
        dist.all_gather(local_sizes, local_size, group=group)
        max_size = torch.stack(local_sizes).max(dim=0).values
        all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

        # 2. If shapes are all the same, then do a simple gather:
        if all_sizes_equal:
            return _simple_gather_all_tensors(result, group, world_size)

        # 3. If not, we need to pad each local tensor to maximum size, gather and then truncate
        pad_dims = []
        pad_by = (max_size - local_size).detach().cpu()
        for val in reversed(pad_by):
            pad_dims.append(0)
            pad_dims.append(val.item())
        result_padded = torch.nn.functional.pad(result, pad_dims)
        gathered_result = [torch.zeros_like(result_padded) for _ in range(world_size)]
        dist.all_gather(gathered_result, result_padded, group)
        for idx, item_size in enumerate(local_sizes):
            slice_param = [slice(dim_size) for dim_size in item_size]
            gathered_result[idx] = gathered_result[idx][slice_param]
        return gathered_result

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


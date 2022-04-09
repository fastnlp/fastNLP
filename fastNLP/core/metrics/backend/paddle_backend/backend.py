from typing import List, Optional, Any

import numpy as np

from fastNLP.core.metrics.backend import Backend
from fastNLP.core.utils.paddle_utils import paddle_to
from fastNLP.core.metrics.utils import AggregateMethodError
from fastNLP.core.utils import is_in_paddle_dist
from fastNLP.core.drivers.paddle_driver.utils import get_device_from_visible
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE

if _NEED_IMPORT_PADDLE:
    import paddle
    import paddle.distributed as dist
    from paddle.fluid.dygraph import parallel_helper

def _simple_gather_all_tensors(result, group: Any, world_size: int) -> List:
    gathered_result = [paddle.zeros_like(result) for _ in range(world_size)]
    dist.all_gather(gathered_result, result, group)
    return gathered_result

class PaddleBackend(Backend):
    def __init__(self):
        super().__init__()
        self._specified = True

    def aggregate(self, tensor, method: str):
        """
        聚集结果，并根据method计算后，返回结果
        """
        if isinstance(tensor, paddle.Tensor):
            if parallel_helper._is_parallel_ctx_initialized():
                if method is None:
                    raise AggregateMethodError(should_have_aggregate_method=True)
                tensor = self._gather_all(tensor)
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

    def create_tensor(self, value: float):
        """
        创建tensor，并且填入value作为值
        """
        tensor = paddle.ones((1,)).fill_(value)
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
        if isinstance(tensor, paddle.Tensor):
            return tensor.cpu().detach().numpy()
        elif isinstance(tensor, np.array):
            return tensor
        else:
            raise ValueError(f"tensor: {tensor} can not convert to ndarray!")

    @staticmethod
    def _gather_all(result, group: Optional[Any] = None) -> List:
        """
        聚合 group 中所有的 result；由于不同 group 中 result 大小不同，因此在适当的时候需要进行 padding
        """
        # TODO check 正确性
        # 有 paddle 那边的 bug，2.3 版本的时候修复了，到时候改一下
        # if group is None:
        #     group = dist.get_group(0)

        world_size = group.nranks if group is not None else dist.get_world_size()
        dist.barrier(group=group)

        # 张量为 标量的情况，简单地gather就好
        if result.ndim == 0:
            return _simple_gather_all_tensors(result, group, world_size)

        # 获得 result 的 shape
        local_size = paddle.to_tensor(result.shape)
        # 将 group 中所有 result 的大小聚合在一起
        local_sizes = []
        dist.all_gather(local_sizes, local_size, group=group)
        # 堆叠后，计算出 shape 每一维度的最大值
        max_size = paddle.stack(local_sizes).max(axis=0)
        all_sizes_equal = all(all(ls == max_size) for ls in local_sizes)

        # 如果所有的结果大小相同，那么可以直接聚合
        if all_sizes_equal:
            return _simple_gather_all_tensors(result, group, world_size)

        # 否则，padding 与最大的张量对齐
        pad_dims = []
        pad_by = (max_size - local_size).detach().cpu()
        for val in reversed(pad_by):
            pad_dims.append(0)
            pad_dims.append(val.item())
        result_padded = paddle.nn.functional.pad(result, pad_dims)
        # 重新进行聚合
        gathered_result = []
        dist.all_gather(gathered_result, result_padded, group)
        for idx, item_size in enumerate(local_sizes):
            slice_param = [slice(dim_size) for dim_size in item_size.tolist()]
            gathered_result[idx] = gathered_result[idx][slice_param]
        return gathered_result

    def move_tensor_to_device(self, tensor, device):
        # TODO 如果在这里处理的话，会不会在别的地方引起bug？
        device = get_device_from_visible(device)
        return paddle_to(tensor, device)


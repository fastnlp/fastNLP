from typing import List, Any

import numpy as np

from fastNLP.core.metrics.backend import Backend
from fastNLP.core.utils.paddle_utils import paddle_to, get_device_from_visible
from fastNLP.core.metrics.utils import AggregateMethodError
from fastNLP.core.drivers.paddle_driver.dist_utils import fastnlp_paddle_all_gather
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
        elif isinstance(tensor, (float, int)):
            return tensor
        else:
            raise ValueError(f"tensor: {tensor} can not convert to ndarray!")

    def move_tensor_to_device(self, tensor, device):
        device = get_device_from_visible(device)
        return paddle_to(tensor, device)

    def all_gather_object(self, obj, group=None) -> List:
        if self.is_distributed():
            obj_list = fastnlp_paddle_all_gather(obj, group=group)
            return obj_list
        return [obj]

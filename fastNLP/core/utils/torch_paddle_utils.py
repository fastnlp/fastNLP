from typing import Any, Optional

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE, _NEED_IMPORT_TORCH

if _NEED_IMPORT_PADDLE:
    import paddle

if _NEED_IMPORT_TORCH:
    import torch

__all__ = [
    "torch_paddle_move_data_to_device",
]

from .utils import apply_to_collection
from .paddle_utils import paddle_to


def torch_paddle_move_data_to_device(batch: Any, device: Optional[str] = None, non_blocking: Optional[bool] = True, 
                                    data_device: Optional[str] = None) -> Any:
    
    r"""
    将数据集合传输到给定设备。只有paddle.Tensor和torch.Tensor对象会被传输到设备中，其余保持不变

    :param batch:
    :param device:
    :param non_blocking:
    :param data_device:
    :return: 相同的集合，但所有包含的张量都驻留在新设备上；
    """

    if device is None:
        if data_device is not None:
            device = data_device
        else:
            return batch

    torch_device = device.replace("gpu", "cuda")
    paddle_device = device.replace("cuda", "gpu")

    def batch_to(data: Any) -> Any:
        if isinstance(data, torch.Tensor):
            data = data.to(torch_device, non_blocking=non_blocking)
        elif isinstance(data, paddle.Tensor):
            data = paddle_to(data, paddle_device)
        
        return data

    return apply_to_collection(batch, dtype=(paddle.Tensor, torch.Tensor), function=batch_to)
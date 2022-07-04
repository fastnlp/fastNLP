

from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    import torch

__all__ = []

def is_torch_tensor_dtype(dtype) -> bool:
    """
    返回当前 dtype 是否是 torch 的 dtype 类型

    :param dtype: 类似与 torch.ones(3).dtype
    :return:
    """
    try:
        return isinstance(dtype, torch.dtype)
    except:
        return False

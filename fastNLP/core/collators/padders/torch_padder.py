
from inspect import isclass
import numpy as np

from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    import torch
    numpy_to_torch_dtype_dict = {
        np.bool_: torch.bool,
        np.uint8: torch.uint8,
        np.int8: torch.int8,
        np.int16: torch.int16,
        np.int32: torch.int32,
        np.int64: torch.int64,
        np.float16: torch.float16,
        np.float32: torch.float32,
        np.float64: torch.float32,  # 这里都统一为到 float32 吧，这是由于 numpy 大部分时候都默认 float64 了
        np.complex64: torch.complex64,
        np.complex128: torch.complex128
    }
    number_to_torch_dtype_dict = {
        float: torch.float32,  # 因为 torch.tensor([1], dtype=float)是torch.float64
        int: torch.int64,
        bool: torch.bool
    }

from .padder import Padder
from .utils import is_number_or_numpy_number, is_number, is_numpy_number_dtype, get_shape, is_numpy_generic_class
from .exceptions import *


def is_torch_tensor(dtype):
    if not isclass(dtype) and isinstance(dtype, torch.dtype):
        return True
    return False


def _get_dtype(ele_dtype, dtype, class_name):
    if not (ele_dtype is not None and (is_number_or_numpy_number(ele_dtype) or is_torch_tensor(ele_dtype))):
        raise EleDtypeUnsupportedError(f"`{class_name}` only supports padding python numbers "
                                       f"or numpy numbers or torch.Tensor but get `{ele_dtype}`.")

    if dtype is not None:
        if not (is_torch_tensor(dtype) or is_number(dtype)):
            raise DtypeUnsupportedError(f"The dtype of `{class_name}` only supports python numbers "
                                        f"or torch.dtype but get `{dtype}`.")
        dtype = number_to_torch_dtype_dict.get(dtype, dtype)
    else:
        if ele_dtype is not None:
            if (is_number(ele_dtype) or is_torch_tensor(ele_dtype)):
                ele_dtype = number_to_torch_dtype_dict.get(ele_dtype, ele_dtype)
                dtype = ele_dtype
            elif is_numpy_number_dtype(ele_dtype): # 存在一个转换的问题了
                dtype = numpy_to_torch_dtype_dict.get(ele_dtype.type)
            elif is_numpy_generic_class(ele_dtype):
                dtype = numpy_to_torch_dtype_dict.get(ele_dtype)

    return dtype


class TorchNumberPadder(Padder):
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        """
        可以将形如 [1, 2, 3] 这类的数据转为 torch.Tensor([1, 2, 3])

        :param pad_val: 该值无意义
        :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 torch.tensor 类型。
        :param dtype: 输出的数据的 dtype 是什么。如 torch.long, torch.float32, int, float 等
        """
        dtype = _get_dtype(ele_dtype, dtype, class_name=self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val, dtype):
        return torch.tensor(batch_field, dtype=dtype)


class TorchSequencePadder(Padder):
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        """
        将类似于 [[1], [1, 2]] 的内容 pad 为 torch.Tensor([[1, 0], [1, 2]]) 可以 pad 多重嵌套的数据。

        :param pad_val: 需要 pad 的值。
        :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 torch.tensor 类型。
        :param dtype: 输出的数据的 dtype 是什么。如 torch.long, torch.float32, int, float 等
        """
        dtype = _get_dtype(ele_dtype, dtype, class_name=self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val, dtype):
        tensor = get_padded_torch_tensor(batch_field, dtype=dtype, pad_val=pad_val)
        return tensor


class TorchTensorPadder(Padder):
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        """
        目前仅支持 [torch.tensor([3, 2], torch.tensor([1])] 类似的

        :param pad_val: 需要 pad 的值。
        :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 torch.tensor 类型。
        :param dtype: 输出的数据的 dtype 是什么。如 torch.long, torch.float32, int, float 等
        """
        dtype = _get_dtype(ele_dtype, dtype, class_name=self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val, dtype):
        shapes = [field.shape for field in batch_field]
        max_shape = [len(batch_field)] + [max(*_) for _ in zip(*shapes)]
        tensor = torch.full(max_shape, fill_value=pad_val, dtype=dtype)
        for i, field in enumerate(batch_field):
            slices = (i, ) + tuple(slice(0, s) for s in shapes[i])
            if isinstance(field, np.ndarray):
                field = torch.from_numpy(field)
            tensor[slices] = field
        return tensor


def fill_tensor(batch_field, padded_batch, dtype):
    """
    将 batch_field 中的值填入到 tensor 中。

    :param batch_field: 需要填充进入 array 中的内容
    :param padded_batch: 待填充的 tensor
    :param dtype: 数据的类别

    :return:
    """
    if padded_batch.ndim == 2:
        for i, content_i in enumerate(batch_field):
            padded_batch[i, :len(content_i)] = torch.tensor(content_i, dtype=dtype)
    elif padded_batch.ndim == 3:
        for i, content_i in enumerate(batch_field):
            for j, content_ii in enumerate(content_i):
                padded_batch[i, j, :len(content_ii)] = torch.tensor(content_ii, dtype=dtype)
    elif padded_batch.ndim == 4:
        try:  # 应该是图像，所以直接应该就 ok 了。
            padded_batch = np.array(batch_field)
        except:
            for i, content_i in enumerate(batch_field):
                for j, content_ii in enumerate(content_i):
                    for k, content_iii in enumerate(content_ii):
                        padded_batch[i, j, k, :len(content_iii)] = torch.tensor(content_iii, dtype=dtype)
    elif padded_batch.ndim == 1:
        padded_batch[:] = torch.tensor(batch_field, dtype=dtype)
    else:
        raise RuntimeError("fastNLP does not support padding for more than 3 dimensions. If you need this, please "
                           "report.")
    return padded_batch


def get_padded_torch_tensor(batch_field, dtype=None, pad_val=0):
    """
    例如:
        [[1,2], [3]] -> torch.LongTensor([[1, 2], [3, 0]])

    :param batch_field: 需要 pad 的对象。需要保证应该是可以进行 pad 的。支持 1d（多为句子长度）/2d（多为文本序列）/3d（多为字符序列）
        /4d（多为图片）。
    :param dtype: 目标类别是什么
    :param pad_val: pad 的 value
    :return:
    """
    shapes = get_shape(batch_field)
    tensor = torch.full(shapes, dtype=dtype, fill_value=pad_val)
    tensor = fill_tensor(batch_field, tensor, dtype=dtype)
    return tensor

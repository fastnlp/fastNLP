__all__ = [
    'OneflowNumberPadder',
    'OneflowSequencePadder',
    'OneflowTensorPadder'
]
from inspect import isclass
import numpy as np

from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW

if _NEED_IMPORT_ONEFLOW:
    import oneflow
    numpy_to_oneflow_dtype_dict = {
        np.bool_: oneflow.bool,
        np.uint8: oneflow.uint8,
        np.int8: oneflow.int8,
        np.int32: oneflow.int32,
        np.int64: oneflow.int64,
        np.float16: oneflow.float16,
        np.float32: oneflow.float32,
        np.float64: oneflow.float32,  # 这里都统一为到 float32 吧，这是由于 numpy 大部分时候都默认 float64 了
    }
    number_to_oneflow_dtype_dict = {
        float: oneflow.float32,  # 因为 oneflow.tensor([1], dtype=float)是oneflow.float64
        int: oneflow.int64,
        bool: oneflow.bool
    }

from .padder import Padder
from .utils import is_number_or_numpy_number, is_number, is_numpy_number_dtype, get_shape, is_numpy_generic_class
from .exceptions import *


def is_oneflow_tensor(dtype):
    """
    判断是否为 oneflow 的 tensor

    :param dtype 数据的 dtype 类型
    """
    if not isclass(dtype) and isinstance(dtype, oneflow.dtype):
        return True
    return False


def _get_dtype(ele_dtype, dtype, class_name):
    """
    用于检测数据的 dtype 类型， 根据内部和外部数据判断。

    :param ele_dtype: 内部数据的类型
    :param dtype:  数据外部类型
    :param class_name: 类的名称
    """
    if not (ele_dtype is None or (is_number_or_numpy_number(ele_dtype) or is_oneflow_tensor(ele_dtype))):
        raise EleDtypeUnsupportedError(f"`{class_name}` only supports padding python numbers "
                                       f"or numpy numbers or oneflow.Tensor but get `{ele_dtype}`.")

    if dtype is not None:
        if not (is_oneflow_tensor(dtype) or is_number(dtype)):
            raise DtypeUnsupportedError(f"The dtype of `{class_name}` only supports python numbers "
                                        f"or oneflow.dtype but get `{dtype}`.")
        dtype = number_to_oneflow_dtype_dict.get(dtype, dtype)
    else:
        if ele_dtype is not None:
            if (is_number(ele_dtype) or is_oneflow_tensor(ele_dtype)):
                ele_dtype = number_to_oneflow_dtype_dict.get(ele_dtype, ele_dtype)
                dtype = ele_dtype
            elif is_numpy_number_dtype(ele_dtype): # 存在一个转换的问题了
                dtype = numpy_to_oneflow_dtype_dict.get(ele_dtype.type)
            elif is_numpy_generic_class(ele_dtype):
                dtype = numpy_to_oneflow_dtype_dict.get(ele_dtype)

    return dtype


class OneflowNumberPadder(Padder):
    """
    可以将形如 [1, 2, 3] 这类的数据转为 oneflow.Tensor([1, 2, 3])

    :param pad_val: 该值无意义
    :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 oneflow.tensor 类型。
    :param dtype: 输出的数据的 dtype 是什么。如 oneflow.long, oneflow.float32, int, float 等
    """
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        dtype = _get_dtype(ele_dtype, dtype, class_name=self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val=0, dtype=None):
        return oneflow.tensor(batch_field, dtype=dtype)


class OneflowSequencePadder(Padder):
    """
    将类似于 [[1], [1, 2]] 的内容 pad 为 oneflow.Tensor([[1, 0], [1, 2]]) 可以 pad 多重嵌套的数据。

    :param pad_val: 需要 pad 的值。
    :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 oneflow.tensor 类型。
    :param dtype: 输出的数据的 dtype 是什么。如 oneflow.long, oneflow.float32, int, float 等
    """
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        dtype = _get_dtype(ele_dtype, dtype, class_name=self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val=0, dtype=None):
        tensor = get_padded_oneflow_tensor(batch_field, dtype=dtype, pad_val=pad_val)
        return tensor


class OneflowTensorPadder(Padder):
    """
    目前支持 [oneflow.tensor([3, 2], oneflow.tensor([1])] 类似的。若内部元素不为 oneflow.tensor ，则必须含有 tolist() 方法。

        >>> OneflowTensorPadder.pad([np.array([3, 4]), np.array([1])], pad_val=-100)
        [[   3.    4.]
         [   1. -100.]]
        >>> OneflowTensorPadder.pad([oneflow.LongTensor([3, 4]), oneflow.LongTensor([1])], pad_val=-100)
        tensor([[   3,    4],
                [   1, -100]])

    :param pad_val: 需要 pad 的值。
    :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 oneflow.tensor 类型。
    :param dtype: 输出的数据的 dtype 是什么。如 oneflow.long, oneflow.float32, int, float 等
    """
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        dtype = _get_dtype(ele_dtype, dtype, class_name=self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val=0, dtype=None):
        device = None
        try:
            if not isinstance(batch_field[0], oneflow.Tensor):
                batch_field = [oneflow.tensor(field.tolist(), dtype=dtype) for field in batch_field]
            else:
                batch_field = [field.to(dtype) for field in batch_field]
                device = batch_field[0].device
            if dtype is None:
                dtype = batch_field[0].dtype
        except AttributeError:
            raise RuntimeError(f"If the field is not a oneflow.Tensor (it is {type(batch_field[0])}), "
                               f"it must have tolist() method.")

        shapes = [field.shape for field in batch_field]
        if len(batch_field) < 2:
            max_shape = [len(batch_field)] + list(shapes[0])
        else:
            max_shape = [len(batch_field)] + [max(*_) for _ in zip(*shapes)]

        tensor = oneflow.full(max_shape, value=pad_val, dtype=dtype, device=device)
        for i, field in enumerate(batch_field):
            slices = (i, ) + tuple(slice(0, s) for s in shapes[i])
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
            padded_batch[i, :len(content_i)] = oneflow.tensor(content_i, dtype=dtype)
    elif padded_batch.ndim == 3:
        for i, content_i in enumerate(batch_field):
            for j, content_ii in enumerate(content_i):
                padded_batch[i, j, :len(content_ii)] = oneflow.tensor(content_ii, dtype=dtype)
    elif padded_batch.ndim == 4:
        try:  # 应该是图像，所以直接应该就 ok 了。
            padded_batch = oneflow.tensor(batch_field)
        except:
            for i, content_i in enumerate(batch_field):
                for j, content_ii in enumerate(content_i):
                    for k, content_iii in enumerate(content_ii):
                        padded_batch[i, j, k, :len(content_iii)] = oneflow.tensor(content_iii, dtype=dtype)
    elif padded_batch.ndim == 1:
        padded_batch[:] = oneflow.tensor(batch_field, dtype=dtype)
    else:
        raise RuntimeError("fastNLP does not support padding for more than 3 dimensions. If you need this, please "
                           "report.")
    return padded_batch


def get_padded_oneflow_tensor(batch_field, dtype=None, pad_val=0):
    """
    例如:
        [[1,2], [3]] -> oneflow.LongTensor([[1, 2], [3, 0]])

    :param batch_field: 需要 pad 的对象。需要保证应该是可以进行 pad 的。支持 1d（多为句子长度）/2d（多为文本序列）/3d（多为字符序列）
        /4d（多为图片）。
    :param dtype: 目标类别是什么
    :param pad_val: pad 的 value
    :return:
    """
    shapes = get_shape(batch_field)
    tensor = oneflow.full(shapes, dtype=dtype, value=pad_val)
    tensor = fill_tensor(batch_field, tensor, dtype=dtype)
    return tensor

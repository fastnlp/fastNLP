__all__ = [
    'NumpyNumberPadder',
    'NumpySequencePadder',
    "NumpyTensorPadder"
]

from numbers import Number
from abc import ABC
from typing import Any, Union
import numpy as np

from .padder import Padder
from .utils import get_padded_numpy_array, is_number_or_numpy_number
from .exceptions import *


def _get_dtype(ele_dtype, dtype, class_name):
    """
    用于检测数据的 dtype 类型， 根据内部和外部数据判断。

    :param ele_dtype 内部数据的类型
    :param dtype  数据外部类型
    :param class_name 类的名称
    """
    if ele_dtype is not None and not is_number_or_numpy_number(ele_dtype):
        raise EleDtypeUnsupportedError(f"`{class_name}` only supports padding python numbers "
                                       f"or numpy numbers but get `{ele_dtype}`.")

    if dtype is None:
        dtype = ele_dtype
    else:
        if not is_number_or_numpy_number(dtype):
            raise DtypeUnsupportedError(f"The dtype of `{class_name}` only supports python numbers "
                                        f"or numpy numbers but get `{dtype}`.")
        dtype = dtype
    return dtype


class NumpyNumberPadder(Padder):
    """
    可以将形如 [1, 2, 3] 这类的数据转为 np.array([1, 2, 3]) 。可以通过:

        >>> NumpyNumberPadder.pad([1, 2, 3])

    使用。

    :param pad_val: 该值无意义
    :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 np.array 类型。
    :param dtype: 输出的数据的 dtype 是什么
    """
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        dtype = _get_dtype(ele_dtype, dtype, self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val=0, dtype=None):
        return np.array(batch_field, dtype=dtype)


class NumpySequencePadder(Padder):
    """
    将类似于 [[1], [1, 2]] 的内容 pad 为 np.array([[1, 0], [1, 2]]) 可以 pad 多重嵌套的数据。
    可以通过以下的方式直接使用:

        >>> NumpySequencePadder.pad([[1], [1, 2]], pad_val=-100, dtype=float)
        [[   1. -100.]
         [   1.    2.]]

    :param pad_val: pad 的值是多少。
    :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 np.array 类型。
    :param dtype: 输出的数据的 dtype 是什么
    """
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        dtype = _get_dtype(ele_dtype, dtype, self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val=0, dtype=None):
        return get_padded_numpy_array(batch_field, dtype=dtype, pad_val=pad_val)


class NumpyTensorPadder(Padder):
    """
    pad 类似于 [np.array([3, 4]), np.array([1])] 的 field 。若内部元素不为 np.ndarray ，则必须含有 tolist() 方法。

        >>> NumpyTensorPadder.pad([np.array([3, 4]), np.array([1])], pad_val=-100)
        [[   3.    4.]
         [   1. -100.]]
    :param pad_val: pad 的值是多少。
    :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 np.array 类型。
    :param dtype: 输出的数据的 dtype 是什么
    """
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        dtype = _get_dtype(ele_dtype, dtype, self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val=0, dtype=None):
        try:
            if not isinstance(batch_field[0], np.ndarray):
                batch_field = [np.array(field.tolist(), dtype=dtype) for field in batch_field]
        except AttributeError:
            raise RuntimeError(f"If the field is not a np.ndarray (it is {type(batch_field[0])}), "
                               f"it must have tolist() method.")

        shapes = [field.shape for field in batch_field]
        if len(batch_field) < 2:
            max_shape = [len(batch_field)] + list(shapes[0])
        else:
            max_shape = [len(batch_field)] + [max(*_) for _ in zip(*shapes)]

        array = np.full(max_shape, fill_value=pad_val, dtype=dtype)
        for i, field in enumerate(batch_field):
            slices = (i, ) + tuple(slice(0, s) for s in shapes[i])
            array[slices] = field
        return array


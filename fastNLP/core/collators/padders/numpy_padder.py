__all__ = [
    'NumpyNumberPadder',
    'NumpySequencePadder',
]

from numbers import Number
from abc import ABC
from typing import Any, Union
import numpy as np

from .padder import Padder
from .utils import get_padded_numpy_array, is_number_or_numpy_number
from .exceptions import *


def _get_dtype(ele_dtype, dtype, class_name):
    if not is_number_or_numpy_number(ele_dtype):
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
    def __init__(self, ele_dtype, pad_val=0, dtype=None):
        dtype = _get_dtype(ele_dtype, dtype, self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val, dtype):
        return np.array(batch_field, dtype=dtype)


class NumpySequencePadder(Padder):
    def __init__(self, ele_dtype, pad_val=0, dtype=None):
        dtype = _get_dtype(ele_dtype, dtype, self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val, dtype):
        return get_padded_numpy_array(batch_field, dtype=dtype, pad_val=pad_val)


class NumpyTensorPadder(Padder):
    def __init__(self, ele_dtype, pad_val=0, dtype=None):
        """
        pad 类似于 [np.array([3, 4], np.array([1])] 的 field

        :param ele_dtype:
        :param pad_val:
        :param dtype:
        """
        dtype = _get_dtype(ele_dtype, dtype, self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val, dtype):
        shapes = [field.shape for field in batch_field]
        max_shape = [len(batch_field)] + [max(*_) for _ in zip(*shapes)]
        array = np.full(max_shape, fill_value=pad_val, dtype=dtype)
        for i, field in enumerate(batch_field):
            slices = (i, ) + tuple(slice(0, s) for s in shapes[i])
            array[slices] = field
        return array


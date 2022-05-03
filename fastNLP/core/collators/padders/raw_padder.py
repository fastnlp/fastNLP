__all__ = [
    "RawNumberPadder",
    "RawSequencePadder",
    "RawTensorPadder"
]

from .padder import Padder
from .utils import is_number, get_padded_numpy_array, is_number_or_numpy_number
from .exceptions import *


def _get_dtype(ele_dtype, dtype, class_name):
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


class RawNumberPadder(Padder):
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        """
        可以将形如 [1, 2, 3] 这类的数据转为 [1, 2, 3] 。实际上该 padder 无意义。

        :param pad_val: 该值无意义
        :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 np.array 类型。
        :param dtype: 输出的数据的 dtype 是什么
        """
        dtype = _get_dtype(ele_dtype, dtype, self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    def __call__(self, batch_field):
        return batch_field

    @staticmethod
    def pad(batch_field, pad_val, dtype):
        raise NotImplementedError()


class RawSequencePadder(Padder):
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        """
        将类似于 [[1], [1, 2]] 的内容 pad 为 [[1, 0], [1, 2]] 。可以 pad 多重嵌套的数据。

        :param pad_val: pad 的值
        :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 np.array 类型。
        :param dtype: 输出的数据的 dtype 是什么
        """
        dtype = _get_dtype(ele_dtype, dtype, self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val, dtype):
        """

        :param batch_field:
        :param pad_val:
        :param dtype: 该参数无意义。
        :return:
        """
        return get_padded_numpy_array(batch_field, dtype=dtype, pad_val=pad_val).tolist()


class RawTensorPadder(Padder):
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        """
        将类似于 [[1], [1, 2]] 的内容 pad 为 [[1, 0], [1, 2]] 。可以 pad 多重嵌套的数据。

        :param pad_val: pad 的值
        :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 np.array 类型。
        :param dtype: 输出的数据的 dtype 是什么
        """
        dtype = _get_dtype(ele_dtype, dtype, self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val, dtype):
        """

        :param batch_field:
        :param pad_val:
        :param dtype: 该参数无意义。
        :return:
        """
        try:
            if not isinstance(batch_field[0], (list, tuple)):
                batch_field = [field.tolist() for field in batch_field]
        except AttributeError:
            raise RuntimeError(f"If the field is not a list or tuple(it is {type(batch_field[0])}), "
                               f"it must have tolist() method.")

        return get_padded_numpy_array(batch_field, dtype=dtype, pad_val=pad_val).tolist()

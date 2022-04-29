

from .padder import Padder
from .utils import get_padded_nest_list, is_number, get_padded_numpy_array
from .exceptions import *


def _get_dtype(ele_dtype, dtype, class_name):
    if is_number(ele_dtype):
        if dtype is None:
            dtype = ele_dtype
        elif not is_number(dtype):
            raise DtypeUnsupportedError(f"The dtype of `{class_name}` can only be None but "
                                        f"get `{dtype}`.")
    else:
        raise EleDtypeUnsupportedError(f"`{class_name}` only supports padding python numbers "
                                       f"but get `{ele_dtype}`.")
    return dtype


class RawNumberPadder(Padder):
    def __init__(self, ele_dtype, pad_val=0, dtype=None):
        dtype = _get_dtype(ele_dtype, dtype, self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    def __call__(self, batch_field):
        return batch_field

    @staticmethod
    def pad(batch_field, pad_val, dtype):
        raise NotImplementedError()


class RawSequencePadder(Padder):
    def __init__(self, ele_dtype, pad_val=0, dtype=None):
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

__all__ = [
    "PaddleNumberPadder",
    "PaddleTensorPadder",
    "PaddleSequencePadder"
]
from inspect import isclass
import numpy as np

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE

if _NEED_IMPORT_PADDLE:
    import paddle
    numpy_to_paddle_dtype_dict = {
        np.bool_: 'bool',
        np.uint8: 'uint8',
        np.int8: "int8",
        np.int16: "int16",
        np.int32: "int32",
        np.int64: "int64",
        np.float16: "float16",
        np.float32: 'float32',
        np.float64: 'float32',  # 这里都统一为到 float32 吧，这是由于 numpy 大部分时候都默认 float64 了
        np.complex64: 'complex64',
        np.complex128: "complex128"
    }
    number_to_paddle_dtype_dict = {
        float: 'float32',  # 因为 paddle.tensor([1], dtype=float)是paddle.float64
        int: 'int64',
        bool: 'bool'
    }

from .padder import Padder
from .utils import is_number_or_numpy_number, is_number, is_numpy_number_dtype, is_numpy_generic_class, \
    get_padded_numpy_array
from .exceptions import *


def is_paddle_tensor(dtype):
    if not isclass(dtype) and isinstance(dtype, paddle.dtype):
        return True

    return False


def is_paddle_dtype_str(dtype):
    try:
        if isinstance(dtype, str) and dtype in {'bool', 'float16', 'uint16', 'float32', 'float64', 'int8',
                    'int16', 'int32', 'int64', 'uint8', 'complex64', 'complex128',
                    u'bool', u'float16', u'uint16', u'float32', u'float64', u'int8',
                    u'int16', u'int32', u'int64', u'uint8', u'complex64',
                    u'complex128'}:
            return True
    except:
        pass
    return False


def _get_dtype(ele_dtype, dtype, class_name):
    if not (ele_dtype is None or is_number_or_numpy_number(ele_dtype) or is_paddle_tensor(ele_dtype) or is_paddle_dtype_str(ele_dtype)):
        raise EleDtypeUnsupportedError(f"`{class_name}` only supports padding python numbers "
                                       f"or numpy numbers or paddle.Tensor but get `{ele_dtype}`.")

    if dtype is not None:
        if not (is_paddle_tensor(dtype) or is_number(dtype) or is_paddle_dtype_str(dtype)):
            raise DtypeUnsupportedError(f"The dtype of `{class_name}` only supports python numbers "
                                        f"or paddle.dtype but get `{dtype}`.")
        dtype = number_to_paddle_dtype_dict.get(dtype, dtype)
    else:
        if (is_number(ele_dtype) or is_paddle_tensor(ele_dtype)):
            ele_dtype = number_to_paddle_dtype_dict.get(ele_dtype, ele_dtype)
            dtype = ele_dtype
        elif is_numpy_number_dtype(ele_dtype): # 存在一个转换的问题了
            dtype = numpy_to_paddle_dtype_dict.get(ele_dtype.type)
        elif is_numpy_generic_class(ele_dtype):
            dtype = numpy_to_paddle_dtype_dict.get(ele_dtype)
        else:
            dtype = ele_dtype

    return dtype


class PaddleNumberPadder(Padder):
    """
    可以将形如 [1, 2, 3] 这类的数据转为 paddle.Tensor([1, 2, 3])

    :param pad_val: 该值无意义
    :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 paddle.tensor 类型。
    :param dtype: 输出的数据的 dtype 是什么。如 int, float, 'int32' 等
    """
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        # 仅当 ele_dtype 是 python number/ numpy number 或者 tensor
        dtype = _get_dtype(ele_dtype, dtype, class_name=self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val=0, dtype=None):
        return paddle.to_tensor(batch_field, dtype=dtype)


class PaddleSequencePadder(Padder):
    """
    将类似于 [[1], [1, 2]] 的内容 pad 为 paddle.Tensor([[1, 0], [1, 2]]) 可以 pad 多重嵌套的数据。

    :param pad_val: pad 的值。
    :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 paddle.tensor 类型。
    :param dtype: 输出的数据的 dtype 是什么。如 int, float, 'int32' 等
    """
    def __init__(self, ele_dtype=None, pad_val=0, dtype=None):
        dtype = _get_dtype(ele_dtype, dtype, class_name=self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val=0, dtype=None):
        tensor = get_padded_paddle_tensor(batch_field, dtype=dtype, pad_val=pad_val)
        return tensor


class PaddleTensorPadder(Padder):
    """
    目前支持 [paddle.tensor([3, 2], paddle.tensor([2, 1])] 类似的，若内部元素不为 paddle.tensor ，则必须含有 tolist() 方法。

    :param pad_val: pad 的值。
    :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 paddle.tensor 类型。
    :param dtype: 输出的数据的 dtype 是什么。如 int, float, 'int32' 等
    """
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        dtype = _get_dtype(ele_dtype, dtype, class_name=self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val=0, dtype=None):
        try:
            if not isinstance(batch_field[0], paddle.Tensor):
                batch_field = [np.array(field.tolist()) for field in batch_field]
            else:
                if dtype is None:
                    dtype = batch_field[0].dtype
        except AttributeError:
            raise RuntimeError(f"If the field is not a paddle.Tensor (it is {type(batch_field[0])}), "
                               f"it must have tolist() method.")

        shapes = [field.shape for field in batch_field]
        max_shape = [len(batch_field)] + [max(*_) for _ in zip(*shapes)]
        if isinstance(batch_field[0], paddle.Tensor):
            array = paddle.full(max_shape, fill_value=pad_val, dtype=dtype)
        else:
            array = np.full(max_shape, fill_value=pad_val, dtype=batch_field[0].dtype)
        for i, field in enumerate(batch_field):
            slices = (i, ) + tuple(slice(0, s) for s in shapes[i])
            array[slices] = field
        tensor = paddle.to_tensor(array, dtype=dtype)
        return tensor


def get_padded_paddle_tensor(batch_field, dtype=None, pad_val=0):
    """
    例如:
        [[1,2], [3]] -> paddle.LongTensor([[1, 2], [3, 0]])

    :param batch_field: 需要 pad 的对象。需要保证应该是可以进行 pad 的。支持 1d（多为句子长度）/2d（多为文本序列）/3d（多为字符序列）
        /4d（多为图片）。
    :param dtype: 目标类别是什么
    :param pad_val: pad 的 value
    :return:
    """
    array = get_padded_numpy_array(batch_field=batch_field, dtype=None, pad_val=pad_val)
    tensor = paddle.to_tensor(array, dtype=dtype)
    return tensor

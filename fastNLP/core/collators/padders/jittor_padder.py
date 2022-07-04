__all__ = [
    'JittorNumberPadder',
    'JittorSequencePadder',
    'JittorTensorPadder'
]

from inspect import isclass
import numpy as np

from fastNLP.envs.imports import _NEED_IMPORT_JITTOR

if _NEED_IMPORT_JITTOR:
    import jittor

    numpy_to_jittor_dtype_dict = {
        np.bool_: 'bool',
        np.uint8: 'uint8',
        np.int8: "int8",
        np.int16: "int16",
        np.int32: "int32",
        np.int64: "int64",
        np.float16: "float16",
        np.float32: 'float32',
        np.float64: 'float32',  # 这里都统一为到 float32 吧，这是由于 numpy 大部分时候都默认 float64 了
    }
    # number_to_jittor_dtype_dict = {
    #     float: 'float32',  # 因为 paddle.tensor([1], dtype=float)是paddle.float64
    #     int: 'int64',
    #     bool: 'bool'
    # }

from .padder import Padder
from .utils import is_number_or_numpy_number, is_number, is_numpy_number_dtype, get_shape, is_numpy_generic_class
from .exceptions import *


def is_jittor_tensor(dtype):
    if not isclass(dtype) and isinstance(dtype, jittor.jittor_core.Var):
        return True
    return False


def is_jittor_dtype_str(dtype):
    """
    判断数据类型是否为 jittor 使用的字符串类型

    :param: dtype 数据类型
    """
    try:
        if isinstance(dtype, str) and dtype in {'bool', 'float16', 'uint16', 'float32', 'float64', 'int8',
                                                'int16', 'int32', 'int64', 'uint8', 'complex64', 'complex128',
                                                u'bool', u'float16', u'uint16', u'float32', u'float64', u'int8',
                                                u'int16', u'int32', u'int64', u'uint8'}:
            return True
    except:
        pass
    return False


def _get_dtype(ele_dtype, dtype, class_name):
    """
    用于检测数据的 dtype 类型， 根据内部和外部数据判断。

    :param ele_dtype 内部数据的类型
    :param dtype  数据外部类型
    :param class_name 类的名称
    """
    if not (ele_dtype is None or (
            is_number_or_numpy_number(ele_dtype) or is_jittor_tensor(ele_dtype) or is_jittor_dtype_str(dtype))):
        raise EleDtypeUnsupportedError(f"`{class_name}` only supports padding python numbers "
                                       f"or numpy numbers or jittor.Var but get `{ele_dtype}`.")

    if dtype is not None:
        if not (is_jittor_tensor(dtype) or is_number(dtype) or is_jittor_dtype_str(dtype)):
            raise DtypeUnsupportedError(f"The dtype of `{class_name}` only supports python numbers "
                                        f"or jittor.dtype but get `{dtype}`.")
    else:
        if is_numpy_generic_class(ele_dtype):
            dtype = numpy_to_jittor_dtype_dict.get(ele_dtype)
        else:
            dtype = ele_dtype

    return dtype


class JittorNumberPadder(Padder):
    """
    可以将形如 ``[1, 2, 3]`` 这类的数据转为 ``jittor.Var([1, 2, 3])``

    :param pad_val: 该值无意义
    :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 :class:`jittor.Var` 类型；
    :param dtype: 输出的数据的 dtype 是什么。如 :class:`jittor.long`, :class:`jittor.float32`, :class:`int`, :class:`float` 等；
    """
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        dtype = _get_dtype(ele_dtype, dtype, class_name=self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val=0, dtype=None):
        """
        :param batch_field 输入的某个 field 的 batch 数据。
        :param pad_val 需要填充的值
        :dtype 数据的类型
        """
        return jittor.Var(np.array(batch_field, dtype=dtype))


class JittorSequencePadder(Padder):
    """
    可以将形如 ``[[1], [1, 2]]`` 这类的数据转为 ``jittor.Var([[1], [1, 2]])``

    :param pad_val: 该值无意义
    :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 :class:`jittor.Var` 类型；
    :param dtype: 输出的数据的 dtype 是什么。如 :class:`jittor.long`, :class:`jittor.float32`, :class:`int`, :class:`float` 等；
    """
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        dtype = _get_dtype(ele_dtype, dtype, class_name=self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val=0, dtype=None):
        """
        :param batch_field: 输入的某个 field 的 batch 数据。
        :param pad_val: 需要填充的值
        :param dtype: 数据的类型
        """
        tensor = get_padded_jittor_tensor(batch_field, dtype=dtype, pad_val=pad_val)
        return tensor


class JittorTensorPadder(Padder):
    def __init__(self, pad_val=0, ele_dtype=None, dtype=None):
        """
        目前支持 ``[jittor.Var([3, 2], jittor.Var([1])]`` 类似的输入。若内部元素不为 :class:`jittor.Var` ，则必须含有 :meth:`tolist` 方法。

        :param pad_val: 需要 pad 的值；
        :param ele_dtype: 用于检测当前 field 的元素类型是否可以转换为 :class:`jittor.Var` 类型；
        :param dtype: 输出的数据的 dtype 是什么。如 :class:`jittor.long`, :class:`jittor.float32`, :class:`int`, :class:`float` 等
        """
        dtype = _get_dtype(ele_dtype, dtype, class_name=self.__class__.__name__)
        super().__init__(pad_val=pad_val, dtype=dtype)

    @staticmethod
    def pad(batch_field, pad_val=0, dtype=None):
        """
        将 ``batch_field`` 数据 转为 :class:`jittor.Var` 并 pad 到相同长度。

        :param batch_field: 输入的某个 field 的 batch 数据。
        :param pad_val: 需要填充的值
        :param dtype: 数据的类型
        """
        try:
            if not isinstance(batch_field[0], jittor.Var):
                batch_field = [jittor.Var(np.array(field.tolist(), dtype=dtype)) for field in batch_field]
        except AttributeError:
            raise RuntimeError(f"If the field is not a jittor.Var (it is {type(batch_field[0])}), "
                               f"it must have tolist() method.")

        shapes = [field.shape for field in batch_field]
        if len(batch_field) < 2:
            max_shape = [len(batch_field)] + list(shapes[0])
        else:
            max_shape = [len(batch_field)] + [max(*_) for _ in zip(*shapes)]

        tensor = jittor.full(max_shape, pad_val, dtype=dtype)
        for i, field in enumerate(batch_field):
            slices = (i,) + tuple(slice(0, s) for s in shapes[i])
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
            padded_batch[i, :len(content_i)] = jittor.Var(np.array(content_i, dtype=dtype))
    elif padded_batch.ndim == 3:
        for i, content_i in enumerate(batch_field):
            for j, content_ii in enumerate(content_i):
                padded_batch[i, j, :len(content_ii)] = jittor.Var(np.array(content_ii, dtype=dtype))
    elif padded_batch.ndim == 4:
        try:  # 应该是图像，所以直接应该就 ok 了。
            padded_batch = jittor.Var(batch_field)
        except:
            for i, content_i in enumerate(batch_field):
                for j, content_ii in enumerate(content_i):
                    for k, content_iii in enumerate(content_ii):
                        padded_batch[i, j, k, :len(content_iii)] = jittor.Var(np.array(content_iii, dtype=dtype))
    elif padded_batch.ndim == 1:
        padded_batch[:] = jittor.Var(np.array(batch_field, dtype=dtype))
    else:
        raise RuntimeError("fastNLP does not support padding for more than 3 dimensions. If you need this, please "
                           "report.")
    return padded_batch


def get_padded_jittor_tensor(batch_field, dtype=None, pad_val=0):
    """
    例如:
        [[1,2], [3]] -> jittor.LongTensor([[1, 2], [3, 0]])

    :param batch_field: 需要 pad 的对象。需要保证应该是可以进行 pad 的。支持 1d（多为句子长度）/2d（多为文本序列）/3d（多为字符序列）
        /4d（多为图片）。
    :param dtype: 目标类别是什么
    :param pad_val: pad 的 value
    :return:
    """
    shapes = get_shape(batch_field)
    tensor = jittor.full(shapes, pad_val, dtype=dtype)
    tensor = fill_tensor(batch_field, tensor, dtype=dtype)
    return tensor

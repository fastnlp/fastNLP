
__all__ = [
    'get_padded_numpy_array'
]


from typing import Sequence, List
import re
from inspect import isclass

import numpy as np
np_str_obj_array_pattern = re.compile(r'[SaUO]')


def get_shape(batch_field:List, shape=None):
    """
    给定 field 返回这个 field pad 完成之后的 shape 。
    例如: [[1, 2, 3], [3]] -> [2, 3]
         [[[1], [2], [3, 4]], [[2, 3, 4]]] -> [2, 3, 3]

    :param batch_field: list，第 0 维一般为 batch 维度。
    :param shape: 无需传入。
    :return:
    """
    if shape is None:
        shape = []
    if isinstance(batch_field, Sequence):
        num_ele = len(batch_field)
        _shape = shape + [num_ele]
        try:
            shapes = []
            if isinstance(batch_field[0], Sequence):
                for _field in batch_field:
                    shapes.append(get_shape(_field, _shape))
                if len(shapes) == 1:
                    max_shape = shapes[0]
                else:
                    max_shape = [max(_) for _ in zip(*shapes)]

                return max_shape
        except IndexError:  # 空的shape
            pass
        return _shape  # 说明是一个空的 sequence
    else:
        return shape


def fill_array(batch_field:List, padded_batch:np.ndarray):
    """
    将 batch_field 中的值填入到 array 中。

    :param batch_field: 需要填充进入 array 中的内容
    :param padded_batch: 待填充的 np.ndarray
    :return:
    """
    if padded_batch.ndim == 2:
        for i, content_i in enumerate(batch_field):
            padded_batch[i, :len(content_i)] = content_i
    elif padded_batch.ndim == 3:
        for i, content_i in enumerate(batch_field):
            for j, content_ii in enumerate(content_i):
                padded_batch[i, j, :len(content_ii)] = content_ii
    elif padded_batch.ndim == 4:
        try:  # 应该是图像，所以直接应该就 ok 了。
            padded_batch = np.array(batch_field)
        except:
            for i, content_i in enumerate(batch_field):
                for j, content_ii in enumerate(content_i):
                    for k, content_iii in enumerate(content_ii):
                        padded_batch[i, j, k, :len(content_iii)] = content_iii
    elif padded_batch.ndim == 1:
        padded_batch[:] = batch_field
    else:
        raise RuntimeError("fastNLP does not support padding for more than 3 dimensions. If you need this, please "
                           "report.")
    return padded_batch


def get_padded_numpy_array(batch_field: List, dtype=None, pad_val=0) -> np.ndarray:
    """
    例如:
        [[1,2], [3]] -> np.array([[1, 2], [3, 0]])

    :param batch_field: 需要 pad 的对象。需要保证应该是可以进行 pad 的。支持 1d（多为句子长度）/2d（多为文本序列）/3d（多为字符序列）
        /4d（多为图片）。
    :param dtype: 目标类别是什么
    :param pad_val: pad 的 value
    :return:
    """
    shapes = get_shape(batch_field)
    array = np.full(shapes, dtype=dtype, fill_value=pad_val)
    array = fill_array(batch_field, array)
    return array


def get_padded_nest_list(batch_field: List, pad_val=0) -> List:
    """
    例如:
        [[1,2], [3]] -> [[1, 2], [3, 0]]

    :param batch_field: 需要 pad 的对象。需要保证应该是可以进行 pad 的。支持 1d（多为句子长度）/2d（多为文本序列）/3d（多为字符序列）
        /4d（多为图片）。
    :param pad_val: pad 的 value
    :return:
    """

    array = get_padded_numpy_array(batch_field, pad_val=pad_val, dtype=None).tolist()
    return array


def is_number_or_numpy_number(dtype):
    """
    判断 dtype 是否是数字类型，或者 numpy 的数字类型。
    is_number_or_numpy_number(type(3))  # True
    is_number_or_numpy_number(type(3.1))  # True
    is_number_or_numpy_number(type('3'))  # False
    is_number_or_numpy_number(type(True))  # True
    is_number_or_numpy_number(type(np.zeros(3)[0]))  # True
    is_number_or_numpy_number(np.zeros(3, dtype=float).dtype)  # True
    is_number_or_numpy_number(np.zeros(3, dtype=int).dtype)  # True
    is_number_or_numpy_number(np.zeros(3, dtype=str).dtype)  # False
    is_number_or_numpy_number(np.array([1, [2]]).dtype)  # False

    :param dtype:
    :return:
    """
    if is_number(dtype):
        return True
    else:
        if isclass(dtype):
            return is_numpy_generic_class(dtype)
        elif isinstance(dtype, np.dtype) and np_str_obj_array_pattern.search(dtype.str) is None:
            return True
    return False


def is_numpy_number_dtype(dtype):
    if not isclass(dtype) and isinstance(dtype, np.dtype) and np_str_obj_array_pattern.search(dtype.str) is None:
        return True
    return False


def is_numpy_generic_class(dtype):
    """
    形如 np.int64，或者 np.zeros(1).dtype.type 的值

    :param dtype:
    :return:
    """
    if isclass(dtype) and issubclass(dtype, np.generic):
        return True
    return False


def is_number(dtype):
    try:
        if dtype in (float, int, complex, bool) and not is_numpy_generic_class(dtype) \
                and not is_numpy_number_dtype(dtype):
            return True
        return False
    except:
        return False



if __name__ == '__main__':
    # a = [[[1]], [1, 2, 3], [3]]
    # a = [[[1], [2], [3, 4]], [[2, 3, 4]]]
    # b = get_padded_nest_list(a)
    # print(type(b[0]))
    # print(b)
    # import torch
    print(is_number(type('a')))
    print(is_number_or_numpy_number(type(3)))  # True
    print(is_number_or_numpy_number(type(3.1)))  # True
    print(is_number_or_numpy_number(type('3'))) # False
    print(is_number_or_numpy_number(type(True)))  # True
    print(is_number_or_numpy_number(type(np.zeros(3)[0])))  # True
    print(is_number_or_numpy_number(np.zeros(3, dtype=float).dtype))  # True
    print(is_number_or_numpy_number(np.zeros(3, dtype=int).dtype))  # True
    print(is_number_or_numpy_number(np.zeros(3, dtype=str).dtype))  # False
    print(is_number_or_numpy_number(np.array([1, [2]]).dtype))  # False


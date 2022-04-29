import pytest
import numpy as np

from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from fastNLP.core.collators.padders.utils import get_shape, get_padded_numpy_array, \
    get_padded_nest_list, is_number_or_numpy_number, is_numpy_number_dtype, is_number


def test_get_shape():
    a = [[1, 2, 3], [3]]
    assert get_shape(a) == [2, 3]

    a = [[[1], [2], [3, 4]], [[2, 3, 4]]]
    assert get_shape(a) == [2, 3, 3]

    a = [[[1], [2], [3, 4]], [[]]]
    assert get_shape(a) == [2, 3, 2]


def test_get_padded_numpy_array():
    a = [[1, 2, 3], [3]]
    a = get_padded_numpy_array(a, dtype=int, pad_val=-1)
    assert a.shape == (2, 3)

    a = [[[1], [2], [3, 4]], [[2, 3, 4]]]
    a = get_padded_numpy_array(a, dtype=int, pad_val=-1)
    assert a.shape == (2, 3, 3)

    a = [[[1], [2], [3, 4]], [[]]]
    a = get_padded_numpy_array(a, dtype=int, pad_val=-1)
    assert a.shape == (2, 3, 2)


def test_get_padded_nest_list():
    a = [[1, 2, 3], [3]]
    a = get_padded_nest_list(a, pad_val=-1)
    assert np.shape(a) == (2, 3)

    a = [[[1], [2], [3, 4]], [[2, 3, 4]]]
    a = get_padded_nest_list(a, pad_val=-1)
    assert np.shape(a) == (2, 3, 3)

    a = [[[1], [2], [3, 4]], [[]]]
    a = get_padded_nest_list(a, pad_val=-1)
    assert np.shape(a) == (2, 3, 2)


def test_is_number_or_numpy_number():
    assert is_number_or_numpy_number(type(3)) is True
    assert is_number_or_numpy_number(type(3.1)) is True
    assert is_number_or_numpy_number(type(True)) is True
    assert is_number_or_numpy_number(type('3')) is False
    assert is_number_or_numpy_number(np.zeros(3).dtype) is True
    assert is_number_or_numpy_number(np.zeros(3, dtype=int).dtype) is True
    assert is_number_or_numpy_number(np.zeros(3, dtype=object).dtype) is False

    if _NEED_IMPORT_TORCH:
        import torch
        dtype = torch.ones(3).dtype
        assert is_number_or_numpy_number(dtype) is False


def test_is_number():
    assert is_number(type(3)) is True
    assert is_number(type(3.1)) is True
    assert is_number(type(True)) is True
    assert is_number(type('3')) is False
    assert is_number(np.zeros(3).dtype) is False
    assert is_number(np.zeros(3, dtype=int).dtype) is False
    assert is_number(np.zeros(3, dtype=object).dtype) is False

    if _NEED_IMPORT_TORCH:
        import torch
        dtype = torch.ones(3).dtype
        assert is_number(dtype) is False


def test_is_numpy_number():
    assert is_numpy_number_dtype(type(3)) is False
    assert is_numpy_number_dtype(type(3.1)) is False
    assert is_numpy_number_dtype(type(True)) is False
    assert is_numpy_number_dtype(type('3')) is False
    assert is_numpy_number_dtype(np.zeros(3).dtype) is True
    assert is_numpy_number_dtype(np.zeros(3, dtype=int).dtype) is True
    assert is_numpy_number_dtype(np.zeros(3, dtype=object).dtype) is False

    if _NEED_IMPORT_TORCH:
        import torch
        dtype = torch.ones(3).dtype
        assert is_numpy_number_dtype(dtype) is False
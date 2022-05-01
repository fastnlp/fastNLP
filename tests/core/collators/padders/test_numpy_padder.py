import numpy as np
import pytest

from fastNLP.core.collators.padders.numpy_padder import NumpyTensorPadder, NumpySequencePadder, NumpyNumberPadder
from fastNLP.core.collators.padders.exceptions import DtypeError
from fastNLP.envs.imports import _NEED_IMPORT_TORCH


class TestNumpyNumberPadder:
    def test_run(self):
        padder = NumpyNumberPadder(ele_dtype=int, dtype=int, pad_val=-1)
        a = [1, 2, 3]
        assert isinstance(padder(a), np.ndarray)
        assert (padder(a) == np.array(a)).sum() == 3


class TestNumpySequencePadder:
    def test_run(self):
        padder = NumpySequencePadder(ele_dtype=int, dtype=int, pad_val=-1)
        a = [[1, 2, 3], [3]]
        a = padder(a)
        shape = np.shape(a)
        assert isinstance(a, np.ndarray)
        assert shape == (2, 3)
        b = np.array([[1, 2, 3], [3, -1, -1]])
        assert (a == b).sum().item() == shape[0]*shape[1]

    def test_dtype_check(self):
        padder = NumpySequencePadder(ele_dtype=np.zeros(3, dtype=np.int8).dtype, dtype=int, pad_val=-1)
        with pytest.raises(DtypeError):
            padder = NumpySequencePadder(ele_dtype=str, dtype=int, pad_val=-1)
        if _NEED_IMPORT_TORCH:
            import torch
            with pytest.raises(DtypeError):
                padder = NumpySequencePadder(ele_dtype=torch.long, dtype=int, pad_val=-1)


class TestNumpyTensorPadder:
    def test_run(self):
        padder = NumpyTensorPadder(ele_dtype=np.zeros(3).dtype, dtype=int, pad_val=-1)
        a = [np.zeros(3), np.zeros(2), np.zeros(0)]
        a = padder(a)
        shape = np.shape(a)
        assert isinstance(a, np.ndarray)
        assert shape == (3, 3)
        b = np.array([[0, 0, 0], [0, 0, -1], [-1, -1, -1]])
        assert (a == b).sum().item() == shape[0]*shape[1]

        a = [np.zeros((3, 2)), np.zeros((2, 2)), np.zeros((1, 1))]
        a = padder(a)
        shape = np.shape(a)
        assert isinstance(a, np.ndarray)
        assert shape == (3, 3, 2)
        b = np.array([[[0, 0], [0, 0], [0, 0]],
                      [[0, 0], [0, 0], [-1, -1]],
                      [[0, -1], [-1, -1], [-1, -1]]])
        assert (a == b).sum().item() == shape[0]*shape[1]*shape[2]

        a = [np.zeros((3, 2)), np.zeros((2, 2)), np.zeros((1, 0))]
        a = padder(a)
        shape = np.shape(a)
        assert isinstance(a, np.ndarray)
        assert shape == (3, 3, 2)
        b = np.array([[[0, 0], [0, 0], [0, 0]],
                      [[0, 0], [0, 0], [-1, -1]],
                      [[-1, -1], [-1, -1], [-1, -1]]])
        assert (a == b).sum().item() == shape[0]*shape[1]*shape[2]

    def test_dtype_check(self):
        padder = NumpyTensorPadder(ele_dtype=np.zeros(3, dtype=np.int8).dtype, dtype=int, pad_val=-1)
        with pytest.raises(DtypeError):
            padder = NumpyTensorPadder(ele_dtype=str, dtype=int, pad_val=-1)
        if _NEED_IMPORT_TORCH:
            import torch
            with pytest.raises(DtypeError):
                padder = NumpyTensorPadder(ele_dtype=torch.long, dtype=int, pad_val=-1)
            with pytest.raises(DtypeError):
                padder = NumpyTensorPadder(ele_dtype=int, dtype=torch.long, pad_val=-1)




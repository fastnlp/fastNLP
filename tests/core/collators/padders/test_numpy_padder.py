import numpy as np
import pytest

from fastNLP.core.collators.padders.numpy_padder import NumpyTensorPadder, NumpySequencePadder, NumpyNumberPadder
from fastNLP.core.collators.padders.exceptions import DtypeError
from fastNLP.envs.imports import _NEED_IMPORT_TORCH


class TestNumpyNumberPadder:
    def test_run(self):
        padder = NumpyNumberPadder(pad_val=-1, ele_dtype=int, dtype=int)
        a = [1, 2, 3]
        assert isinstance(padder(a), np.ndarray)
        assert (padder(a) == np.array(a)).sum() == 3


@pytest.mark.torch
class TestNumpySequencePadder:
    def test_run(self):
        padder = NumpySequencePadder(pad_val=-1, ele_dtype=int, dtype=int)
        a = [[1, 2, 3], [3]]
        a = padder(a)
        shape = np.shape(a)
        assert isinstance(a, np.ndarray)
        assert shape == (2, 3)
        b = np.array([[1, 2, 3], [3, -1, -1]])
        assert (a == b).sum().item() == shape[0]*shape[1]

    def test_dtype_check(self):
        padder = NumpySequencePadder(pad_val=-1, ele_dtype=np.zeros(3, dtype=np.int8).dtype, dtype=int)
        with pytest.raises(DtypeError):
            padder = NumpySequencePadder(pad_val=-1, ele_dtype=str, dtype=int)
        if _NEED_IMPORT_TORCH:
            import torch
            with pytest.raises(DtypeError):
                padder = NumpySequencePadder(pad_val=-1, ele_dtype=torch.long, dtype=int)


class TestNumpyTensorPadder:
    def test_run(self):
        padder = NumpyTensorPadder(pad_val=-1, ele_dtype=np.zeros(3).dtype, dtype=int)
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
        padder = NumpyTensorPadder(pad_val=-1, ele_dtype=np.zeros(3, dtype=np.int8).dtype, dtype=int)
        with pytest.raises(DtypeError):
            padder = NumpyTensorPadder(pad_val=-1, ele_dtype=str, dtype=int)
        if _NEED_IMPORT_TORCH:
            import torch
            with pytest.raises(DtypeError):
                padder = NumpyTensorPadder(pad_val=-1, ele_dtype=torch.long, dtype=int)
            with pytest.raises(DtypeError):
                padder = NumpyTensorPadder(pad_val=-1, ele_dtype=int, dtype=torch.long)




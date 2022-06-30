import numpy as np
import pytest

from fastNLP.core.collators.padders.oneflow_padder import OneflowTensorPadder, OneflowSequencePadder, OneflowNumberPadder
from fastNLP.core.collators.padders.exceptions import DtypeError
from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW

if _NEED_IMPORT_ONEFLOW:
    import oneflow


@pytest.mark.oneflow
class TestOneflowNumberPadder:
    def test_run(self):
        padder = OneflowNumberPadder(pad_val=-1, ele_dtype=int, dtype=int)
        a = [1, 2, 3]
        t_a = padder(a)
        assert isinstance(t_a, oneflow.Tensor)
        assert (t_a == oneflow.LongTensor(a)).sum() == 3


@pytest.mark.oneflow
class TestOneflowSequencePadder:
    def test_run(self):
        padder = OneflowSequencePadder(pad_val=-1, ele_dtype=int, dtype=int)
        a = [[1, 2, 3], [3]]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, oneflow.Tensor)
        assert tuple(shape) == (2, 3)
        b = oneflow.LongTensor([[1, 2, 3], [3, -1, -1]])
        assert (a == b).sum().item() == shape[0]*shape[1]

    def test_dtype_check(self):
        padder = OneflowSequencePadder(pad_val=-1, ele_dtype=np.zeros(3, dtype=np.int8).dtype, dtype=int)
        with pytest.raises(DtypeError):
            padder = OneflowSequencePadder(pad_val=-1, ele_dtype=str, dtype=int)
        padder = OneflowSequencePadder(pad_val=-1, ele_dtype=oneflow.long, dtype=int)
        padder = OneflowSequencePadder(pad_val=-1, ele_dtype=np.int8, dtype=None)
        a = padder([[1], [2, 322]])
        assert (a>67).sum()==0  # 因为int8的范围为-67 - 66
        padder = OneflowSequencePadder(pad_val=-1, ele_dtype=np.zeros(2).dtype, dtype=None)


@pytest.mark.oneflow
class TestOneflowTensorPadder:
    def test_run(self):
        padder = OneflowTensorPadder(pad_val=-1, ele_dtype=oneflow.zeros(3).dtype, dtype=int)
        a = [oneflow.zeros(3), oneflow.zeros(2), oneflow.zeros(0)]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, oneflow.Tensor)
        assert tuple(shape) == (3, 3)
        b = oneflow.LongTensor([[0, 0, 0], [0, 0, -1], [-1, -1, -1]])
        assert (a == b).sum().item() == shape[0]*shape[1]

        a = [oneflow.zeros((3, 2)), oneflow.zeros((2, 2)), oneflow.zeros((1, 2))]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, oneflow.Tensor)
        assert tuple(shape) == (3, 3, 2)
        b = oneflow.LongTensor([[[0, 0], [0, 0], [0, 0]],
                      [[0, 0], [0, 0], [-1, -1]],
                      [[0, 0], [-1, -1], [-1, -1]]])
        assert (a == b).sum().item() == shape[0]*shape[1]*shape[2]

        a = [oneflow.zeros((3, 2)), oneflow.zeros((2, 2)), oneflow.zeros((1, 1))]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, oneflow.Tensor)
        assert tuple(shape) == (3, 3, 2)
        b = oneflow.LongTensor([[[0, 0], [0, 0], [0, 0]],
                              [[0, 0], [0, 0], [-1, -1]],
                              [[0, -1], [-1, -1], [-1, -1]]])
        assert (a == b).sum().item() == shape[0]*shape[1]*shape[2]

        padder = OneflowTensorPadder(pad_val=-1, ele_dtype=oneflow.zeros(3).dtype, dtype=int)
        a = [oneflow.zeros((3, 2)), oneflow.zeros((2, 2)), oneflow.zeros((1, 0))]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, oneflow.Tensor)
        assert tuple(shape) == (3, 3, 2)
        b = oneflow.LongTensor([[[0, 0], [0, 0], [0, 0]],
                              [[0, 0], [0, 0], [-1, -1]],
                              [[-1, -1], [-1, -1], [-1, -1]]])
        assert (a == b).sum().item() == shape[0]*shape[1]*shape[2]

        padder = OneflowTensorPadder(pad_val=-1, ele_dtype=oneflow.zeros(3).dtype, dtype=None)
        a = [np.zeros((3, 2)), np.zeros((2, 2)), np.zeros((1, 0))]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, oneflow.Tensor)
        assert tuple(shape) == (3, 3, 2)
        b = oneflow.FloatTensor([[[0, 0], [0, 0], [0, 0]],
                              [[0, 0], [0, 0], [-1, -1]],
                              [[-1, -1], [-1, -1], [-1, -1]]])
        assert (a == b).sum().item() == shape[0]*shape[1]*shape[2]

    def test_dtype_check(self):
        padder = OneflowTensorPadder(pad_val=-1, ele_dtype=np.zeros(3, dtype=np.int8).dtype, dtype=int)
        with pytest.raises(DtypeError):
            padder = OneflowTensorPadder(pad_val=-1, ele_dtype=str, dtype=int)
        padder = OneflowTensorPadder(pad_val=-1, ele_dtype=oneflow.long, dtype=int)
        padder = OneflowTensorPadder(pad_val=-1, ele_dtype=int, dtype=oneflow.long)


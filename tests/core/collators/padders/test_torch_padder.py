import numpy as np
import pytest

from fastNLP.core.collators.padders.torch_padder import TorchTensorPadder, TorchSequencePadder, TorchNumberPadder
from fastNLP.core.collators.padders.exceptions import DtypeError
from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    import torch


@pytest.mark.torch
class TestTorchNumberPadder:
    def test_run(self):
        padder = TorchNumberPadder(pad_val=-1, ele_dtype=int, dtype=int)
        a = [1, 2, 3]
        t_a = padder(a)
        assert isinstance(t_a, torch.Tensor)
        assert (t_a == torch.LongTensor(a)).sum() == 3


@pytest.mark.torch
class TestTorchSequencePadder:
    def test_run(self):
        padder = TorchSequencePadder(pad_val=-1, ele_dtype=int, dtype=int)
        a = [[1, 2, 3], [3]]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, torch.Tensor)
        assert tuple(shape) == (2, 3)
        b = torch.LongTensor([[1, 2, 3], [3, -1, -1]])
        assert (a == b).sum().item() == shape[0]*shape[1]

    def test_dtype_check(self):
        padder = TorchSequencePadder(pad_val=-1, ele_dtype=np.zeros(3, dtype=np.int8).dtype, dtype=int)
        with pytest.raises(DtypeError):
            padder = TorchSequencePadder(pad_val=-1, ele_dtype=str, dtype=int)
        padder = TorchSequencePadder(pad_val=-1, ele_dtype=torch.long, dtype=int)
        padder = TorchSequencePadder(pad_val=-1, ele_dtype=np.int8, dtype=None)
        a = padder([[1], [2, 322]])
        assert (a>67).sum()==0  # 因为int8的范围为-67 - 66
        padder = TorchSequencePadder(pad_val=-1, ele_dtype=np.zeros(2).dtype, dtype=None)


@pytest.mark.torch
class TestTorchTensorPadder:
    def test_run(self):
        padder = TorchTensorPadder(pad_val=-1, ele_dtype=torch.zeros(3).dtype, dtype=int)
        a = [torch.zeros(3), torch.zeros(2), torch.zeros(0)]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, torch.Tensor)
        assert tuple(shape) == (3, 3)
        b = torch.LongTensor([[0, 0, 0], [0, 0, -1], [-1, -1, -1]])
        assert (a == b).sum().item() == shape[0]*shape[1]

        a = [torch.zeros((3, 2)), torch.zeros((2, 2)), torch.zeros((1, 2))]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, torch.Tensor)
        assert tuple(shape) == (3, 3, 2)
        b = torch.LongTensor([[[0, 0], [0, 0], [0, 0]],
                      [[0, 0], [0, 0], [-1, -1]],
                      [[0, 0], [-1, -1], [-1, -1]]])
        assert (a == b).sum().item() == shape[0]*shape[1]*shape[2]

        a = [torch.zeros((3, 2)), torch.zeros((2, 2)), torch.zeros((1, 1))]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, torch.Tensor)
        assert tuple(shape) == (3, 3, 2)
        b = torch.LongTensor([[[0, 0], [0, 0], [0, 0]],
                              [[0, 0], [0, 0], [-1, -1]],
                              [[0, -1], [-1, -1], [-1, -1]]])
        assert (a == b).sum().item() == shape[0]*shape[1]*shape[2]

        padder = TorchTensorPadder(pad_val=-1, ele_dtype=torch.zeros(3).dtype, dtype=int)
        a = [torch.zeros((3, 2)), torch.zeros((2, 2)), torch.zeros((1, 0))]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, torch.Tensor)
        assert tuple(shape) == (3, 3, 2)
        b = torch.LongTensor([[[0, 0], [0, 0], [0, 0]],
                              [[0, 0], [0, 0], [-1, -1]],
                              [[-1, -1], [-1, -1], [-1, -1]]])
        assert (a == b).sum().item() == shape[0]*shape[1]*shape[2]

        padder = TorchTensorPadder(pad_val=-1, ele_dtype=torch.zeros(3).dtype, dtype=None)
        a = [np.zeros((3, 2)), np.zeros((2, 2)), np.zeros((1, 0))]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, torch.Tensor)
        assert tuple(shape) == (3, 3, 2)
        b = torch.FloatTensor([[[0, 0], [0, 0], [0, 0]],
                              [[0, 0], [0, 0], [-1, -1]],
                              [[-1, -1], [-1, -1], [-1, -1]]])
        assert (a == b).sum().item() == shape[0]*shape[1]*shape[2]

    def test_dtype_check(self):
        padder = TorchTensorPadder(pad_val=-1, ele_dtype=np.zeros(3, dtype=np.int8).dtype, dtype=int)
        with pytest.raises(DtypeError):
            padder = TorchTensorPadder(pad_val=-1, ele_dtype=str, dtype=int)
        padder = TorchTensorPadder(pad_val=-1, ele_dtype=torch.long, dtype=int)
        padder = TorchTensorPadder(pad_val=-1, ele_dtype=int, dtype=torch.long)




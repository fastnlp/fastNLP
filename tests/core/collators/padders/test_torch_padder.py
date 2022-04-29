import numpy as np
import pytest

from fastNLP.core.collators.padders.torch_padder import TorchTensorPadder, TorchSequencePadder, TorchNumberPadder
from fastNLP.core.collators.padders.exceptions import DtypeError
from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    import torch


class TestTorchNumberPadder:
    def test_run(self):
        padder = TorchNumberPadder(ele_dtype=int, dtype=int, pad_val=-1)
        a = [1, 2, 3]
        t_a = padder(a)
        assert isinstance(t_a, torch.Tensor)
        assert (t_a == torch.LongTensor(a)).sum() == 3


class TestTorchSequencePadder:
    def test_run(self):
        padder = TorchSequencePadder(ele_dtype=int, dtype=int, pad_val=-1)
        a = [[1, 2, 3], [3]]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, torch.Tensor)
        assert tuple(shape) == (2, 3)
        b = torch.LongTensor([[1, 2, 3], [3, -1, -1]])
        assert (a == b).sum().item() == shape[0]*shape[1]

    def test_dtype_check(self):
        padder = TorchSequencePadder(ele_dtype=np.zeros(3, dtype=np.int8).dtype, dtype=int, pad_val=-1)
        with pytest.raises(DtypeError):
            padder = TorchSequencePadder(ele_dtype=str, dtype=int, pad_val=-1)
        padder = TorchSequencePadder(ele_dtype=torch.long, dtype=int, pad_val=-1)
        padder = TorchSequencePadder(ele_dtype=np.int8, dtype=None, pad_val=-1)
        a = padder([[1], [2, 322]])
        assert (a>67).sum()==0  # 因为int8的范围为-67 - 66
        padder = TorchSequencePadder(ele_dtype=np.zeros(2).dtype, dtype=None, pad_val=-1)



class TestTorchTensorPadder:
    def test_run(self):
        padder = TorchTensorPadder(ele_dtype=torch.zeros(3).dtype, dtype=int, pad_val=-1)
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

        padder = TorchTensorPadder(ele_dtype=torch.zeros(3).dtype, dtype=int, pad_val=-1)
        a = [torch.zeros((3, 2)), torch.zeros((2, 2)), torch.zeros((1, 0))]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, torch.Tensor)
        assert tuple(shape) == (3, 3, 2)
        b = torch.LongTensor([[[0, 0], [0, 0], [0, 0]],
                              [[0, 0], [0, 0], [-1, -1]],
                              [[-1, -1], [-1, -1], [-1, -1]]])
        assert (a == b).sum().item() == shape[0]*shape[1]*shape[2]

        padder = TorchTensorPadder(ele_dtype=torch.zeros(3).dtype, dtype=None, pad_val=-1)
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
        padder = TorchTensorPadder(ele_dtype=np.zeros(3, dtype=np.int8).dtype, dtype=int, pad_val=-1)
        with pytest.raises(DtypeError):
            padder = TorchTensorPadder(ele_dtype=str, dtype=int, pad_val=-1)
        padder = TorchTensorPadder(ele_dtype=torch.long, dtype=int, pad_val=-1)
        padder = TorchTensorPadder(ele_dtype=int, dtype=torch.long, pad_val=-1)




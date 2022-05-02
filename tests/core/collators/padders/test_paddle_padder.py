import numpy as np
import pytest

from fastNLP.core.collators.padders.paddle_padder import paddleTensorPadder, paddleSequencePadder, paddleNumberPadder
from fastNLP.core.collators.padders.exceptions import DtypeError
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE

if _NEED_IMPORT_PADDLE:
    import paddle


@pytest.mark.paddle
class TestpaddleNumberPadder:
    def test_run(self):
        padder = paddleNumberPadder(ele_dtype=int, dtype=int, pad_val=-1)
        a = [1, 2, 3]
        t_a = padder(a)
        assert isinstance(t_a, paddle.Tensor)
        assert (t_a == paddle.to_tensor(a, dtype='int64')).sum() == 3


@pytest.mark.paddle
class TestpaddleSequencePadder:
    def test_run(self):
        padder = paddleSequencePadder(ele_dtype=int, dtype=int, pad_val=-1)
        a = [[1, 2, 3], [3]]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, paddle.Tensor)
        assert tuple(shape) == (2, 3)
        b = paddle.to_tensor([[1, 2, 3], [3, -1, -1]], dtype='int64')
        assert (a == b).sum().item() == shape[0]*shape[1]

    def test_dtype_check(self):
        padder = paddleSequencePadder(ele_dtype=np.zeros(3, dtype=np.int32).dtype, dtype=int, pad_val=-1)
        with pytest.raises(DtypeError):
            padder = paddleSequencePadder(ele_dtype=str, dtype=int, pad_val=-1)
        padder = paddleSequencePadder(ele_dtype='int64', dtype=int, pad_val=-1)
        padder = paddleSequencePadder(ele_dtype=np.int32, dtype=None, pad_val=-1)
        a = padder([[1], [2, 322]])
        # assert (a>67).sum()==0  # 因为int8的范围为-67 - 66
        padder = paddleSequencePadder(ele_dtype=np.zeros(2).dtype, dtype=None, pad_val=-1)


@pytest.mark.paddle
class TestpaddleTensorPadder:
    def test_run(self):
        padder = paddleTensorPadder(ele_dtype=paddle.zeros((3,)).dtype, dtype=paddle.zeros((3,)).dtype, pad_val=-1)
        a = [paddle.zeros((3,)), paddle.zeros((2,))]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, paddle.Tensor)
        assert tuple(shape) == (2, 3)
        b = paddle.to_tensor([[0, 0, 0], [0, 0, -1]], dtype='int64')
        assert (a == b).sum().item() == shape[0]*shape[1]

        a = [paddle.zeros((3, 2)), paddle.zeros((2, 2)), paddle.zeros((1, 2))]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, paddle.Tensor)
        assert tuple(shape) == (3, 3, 2)
        b = paddle.to_tensor([[[0, 0], [0, 0], [0, 0]],
                      [[0, 0], [0, 0], [-1, -1]],
                      [[0, 0], [-1, -1], [-1, -1]]], dtype='int64')
        assert (a == b).sum().item() == shape[0]*shape[1]*shape[2]

        a = [paddle.zeros((3, 2)), paddle.zeros((2, 2)), paddle.zeros((1, 1))]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, paddle.Tensor)
        assert tuple(shape) == (3, 3, 2)
        b = paddle.to_tensor([[[0, 0], [0, 0], [0, 0]],
                              [[0, 0], [0, 0], [-1, -1]],
                              [[0, -1], [-1, -1], [-1, -1]]])
        assert (a == b).sum().item() == shape[0]*shape[1]*shape[2]

        padder = paddleTensorPadder(ele_dtype=paddle.zeros((3, )).dtype, dtype=paddle.zeros((3, )).dtype, pad_val=-1)
        a = [paddle.zeros((3, 2)), paddle.zeros((2, 2))]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, paddle.Tensor)
        assert tuple(shape) == (2, 3, 2)
        b = paddle.to_tensor([[[0, 0], [0, 0], [0, 0]],
                              [[0, 0], [0, 0], [-1, -1]],
                              ])
        assert (a == b).sum().item() == shape[0]*shape[1]*shape[2]

        padder = paddleTensorPadder(ele_dtype=paddle.zeros((3, 2)).dtype, dtype=None, pad_val=-1)
        a = [np.zeros((3, 2), dtype=np.float32), np.zeros((2, 2), dtype=np.float32)]
        a = padder(a)
        shape = a.shape
        assert isinstance(a, paddle.Tensor)
        assert tuple(shape) == (2, 3, 2)
        b = paddle.to_tensor([[[0, 0], [0, 0], [0, 0]],
                              [[0, 0], [0, 0], [-1, -1]]], dtype='float32')
        assert (a == b).sum().item() == shape[0]*shape[1]*shape[2]

    def test_dtype_check(self):
        padder = paddleTensorPadder(ele_dtype=np.zeros(3, dtype=np.int8).dtype, dtype=int, pad_val=-1)
        with pytest.raises(DtypeError):
            padder = paddleTensorPadder(ele_dtype=str, dtype=int, pad_val=-1)
        padder = paddleTensorPadder(ele_dtype='int64', dtype=int, pad_val=-1)
        padder = paddleTensorPadder(ele_dtype=int, dtype='int64', pad_val=-1)

    def test_v1(self):
        print(paddle.zeros((3, )).dtype)

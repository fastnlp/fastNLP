import pytest
import numpy as np
from fastNLP.core.utils.seq_len_to_mask import seq_len_to_mask
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR, _NEED_IMPORT_PADDLE, _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch

if _NEED_IMPORT_PADDLE:
    import paddle

if _NEED_IMPORT_JITTOR:
    import jittor


class TestSeqLenToMask:

    def evaluate_mask_seq_len(self, seq_len, mask):
        max_len = int(max(seq_len))
        for i in range(len(seq_len)):
            length = seq_len[i]
            mask_i = mask[i]
            for j in range(max_len):
                assert mask_i[j] == (j<length), (i, j, length)

    def test_numpy_seq_len(self):
        # 测试能否转换numpy类型的seq_len
        # 1. 随机测试
        seq_len = np.random.randint(1, 10, size=(2, ))
        mask = seq_len_to_mask(seq_len)
        max_len = seq_len.max()
        assert max_len == mask.shape[1]
        print(mask)
        print(seq_len)
        self.evaluate_mask_seq_len(seq_len, mask)

        # 2. 异常检测
        seq_len = np.random.randint(10, size=(10, 1))

        with pytest.raises(AssertionError):
            mask = seq_len_to_mask(seq_len)

        # 3. pad到指定长度
        seq_len = np.random.randint(1, 10, size=(10,))
        mask = seq_len_to_mask(seq_len, 100)
        assert 100 == mask.shape[1]

    @pytest.mark.torch
    def test_pytorch_seq_len(self):
        # 1. 随机测试
        seq_len = torch.randint(1, 10, size=(10, ))
        max_len = seq_len.max()
        mask = seq_len_to_mask(seq_len)
        assert max_len == mask.shape[1]
        self.evaluate_mask_seq_len(seq_len.tolist(), mask)

        # 2. 异常检测
        seq_len = torch.randn(3, 4)
        with pytest.raises(AssertionError):
            mask = seq_len_to_mask(seq_len)

        # 3. pad到指定长度
        seq_len = torch.randint(1, 10, size=(10, ))
        mask = seq_len_to_mask(seq_len, 100)
        assert 100 == mask.size(1)

    @pytest.mark.paddle
    def test_paddle_seq_len(self):
        # 1. 随机测试
        seq_len = paddle.randint(1, 10, shape=(10,))
        max_len = seq_len.max()
        mask = seq_len_to_mask(seq_len)
        assert max_len == mask.shape[1]
        self.evaluate_mask_seq_len(seq_len.tolist(), mask)

        # 2. 异常检测
        seq_len = paddle.randn((3, 4))
        with pytest.raises(AssertionError):
            mask = seq_len_to_mask(seq_len)

        # 3. pad到指定长度
        seq_len = paddle.randint(1, 10, size=(10,))
        mask = seq_len_to_mask(seq_len, 100)
        assert 100 == mask.shape[1]

    @pytest.mark.jittor
    def test_jittor_seq_len(self):
        # 1. 随机测试
        seq_len = jittor.randint(1, 10, shape=(10,))
        max_len = seq_len.max()
        mask = seq_len_to_mask(seq_len)
        assert max_len == mask.shape[1]
        self.evaluate_mask_seq_len(seq_len.tolist(), mask)

        # 2. 异常检测
        seq_len = jittor.randn(3, 4)
        with pytest.raises(AssertionError):
            mask = seq_len_to_mask(seq_len)

        # 3. pad到指定长度
        seq_len = jittor.randint(1, 10, shape=(10,))
        mask = seq_len_to_mask(seq_len, 100)
        assert 100 == mask.shape[1]

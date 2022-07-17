from typing import Optional

import numpy as np
from ...envs.imports import _NEED_IMPORT_JITTOR, _NEED_IMPORT_TORCH, _NEED_IMPORT_PADDLE, _NEED_IMPORT_ONEFLOW
from .paddle_utils import paddle_to


if _NEED_IMPORT_TORCH:
    import torch

if _NEED_IMPORT_PADDLE:
    import paddle

if _NEED_IMPORT_JITTOR:
    import jittor

if _NEED_IMPORT_ONEFLOW:
    import oneflow


def seq_len_to_mask(seq_len, max_len: Optional[int]=None):
    r"""

    将一个表示 ``sequence length`` 的一维数组转换为二维的 ``mask`` ，不包含的位置为 **0**。

    .. code-block::

        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.size())
        torch.Size([14, 15])
        >>> seq_len = np.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len)
        >>> print(mask.shape)
        (14, 15)
        >>> seq_len = torch.arange(2, 16)
        >>> mask = seq_len_to_mask(seq_len, max_len=100)
        >>>print(mask.size())
        torch.Size([14, 100])

    :param seq_len: 大小为 ``(B,)`` 的长度序列；
    :param int max_len: 将长度补齐或截断到 ``max_len``。默认情况（为 ``None``）使用的是 ``seq_len`` 中最长的长度；
        但在 :class:`torch.nn.DataParallel` 等分布式的场景下可能不同卡的 ``seq_len`` 会有区别，所以需要传入
        ``max_len`` 使得 ``mask`` 的补齐或截断到该长度。
    :return: 大小为 ``(B, max_len)`` 的 ``mask``， 元素类型为 ``bool`` 或 ``uint8``
    """
    max_len = int(max_len) if max_len is not None else int(seq_len.max())

    if isinstance(seq_len, np.ndarray):
        assert seq_len.ndim == 1, f"seq_len can only have one dimension, got {seq_len.ndim}."
        broad_cast_seq_len = np.tile(np.arange(max_len), (len(seq_len), 1))
        mask = broad_cast_seq_len < seq_len.reshape(-1, 1)
        return mask

    try:  # 尝试是否是 torch
        if isinstance(seq_len, torch.Tensor):
            assert seq_len.ndim == 1, f"seq_len can only have one dimension, got {seq_len.ndim == 1}."
            batch_size = seq_len.shape[0]
            broad_cast_seq_len = torch.arange(max_len).expand(batch_size, -1).to(seq_len)
            mask = broad_cast_seq_len < seq_len.unsqueeze(1)
            return mask
    except NameError as e:
        pass

    try:
        if isinstance(seq_len, paddle.Tensor):
            assert seq_len.ndim == 1, f"seq_len can only have one dimension, got {seq_len.ndim == 1}."
            batch_size = seq_len.shape[0]
            broad_cast_seq_len = paddle.arange(max_len).expand((batch_size, -1))
            broad_cast_seq_len = paddle_to(broad_cast_seq_len, device=seq_len.place)
            mask = broad_cast_seq_len < seq_len.unsqueeze(1)
            return mask
    except NameError as e:
        pass

    try:
        if isinstance(seq_len, jittor.Var):
            assert seq_len.ndim == 1, f"seq_len can only have one dimension, got {seq_len.ndim == 1}."
            batch_size = seq_len.shape[0]
            broad_cast_seq_len = jittor.arange(max_len).reshape(1, max_len).expand(batch_size, -1)
            mask = broad_cast_seq_len < seq_len.unsqueeze(1)
            return mask
    except NameError as e:
        pass

    try:
        if isinstance(seq_len, oneflow.Tensor):
            assert seq_len.ndim == 1, f"seq_len can only have one dimension, got {seq_len.ndim == 1}."
            batch_size = seq_len.shape[0]
            broad_cast_seq_len = oneflow.arange(max_len).expand(batch_size, -1).to(seq_len)
            mask = broad_cast_seq_len < seq_len.unsqueeze(1)
            return mask
    except NameError as e:
        pass

    raise TypeError("seq_len_to_mask function only supports numpy.ndarray, torch.Tensor, paddle.Tensor, "
                    f"jittor.Var and oneflow.Tensor, but got {type(seq_len)}")
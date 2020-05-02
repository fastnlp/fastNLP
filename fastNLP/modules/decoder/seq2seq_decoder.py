# coding=utf-8
__all__ = [
    "TransformerPast",
    "Past",
    "Decoder"
]
import torch
from torch import nn
import abc
import torch.nn.functional as F
from ...embeddings import StaticEmbedding
import numpy as np
from typing import Union, Tuple
from ...embeddings.utils import get_embeddings
from torch.nn import LayerNorm
import math


class Past:
    def __init__(self):
        pass

    @abc.abstractmethod
    def num_samples(self):
        pass

    @abc.abstractmethod
    def reorder_past(self, indices: torch.LongTensor):
        """
        根据indices中的index，将past的中状态置为正确的顺序。inplace改变

        :param torch.LongTensor indices:
        :param Past past:
        :return:
        """
        raise NotImplemented


class TransformerPast(Past):
    def __init__(self, encoder_outputs: torch.Tensor = None, encoder_mask: torch.Tensor = None,
                 num_decoder_layer: int = 6):
        """

        :param encoder_outputs: (batch,src_seq_len,dim)
        :param encoder_mask: (batch,src_seq_len)
        :param encoder_key: list of (batch, src_seq_len, dim)
        :param encoder_value:
        :param decoder_prev_key:
        :param decoder_prev_value:
        """
        super().__init__()
        self.encoder_outputs = encoder_outputs
        self.encoder_mask = encoder_mask
        self.encoder_key = [None] * num_decoder_layer
        self.encoder_value = [None] * num_decoder_layer
        self.decoder_prev_key = [None] * num_decoder_layer
        self.decoder_prev_value = [None] * num_decoder_layer

    def num_samples(self):
        if self.encoder_outputs is not None:
            return self.encoder_outputs.size(0)
        return None

    def _reorder_state(self, state, indices):
        if type(state) == torch.Tensor:
            state = state.index_select(index=indices, dim=0)
        elif type(state) == list:
            for i in range(len(state)):
                assert state[i] is not None
                state[i] = state[i].index_select(index=indices, dim=0)
        else:
            raise ValueError('State does not support other format')

        return state

    def reorder_past(self, indices: torch.LongTensor):
        self.encoder_outputs = self._reorder_state(self.encoder_outputs, indices)
        self.encoder_mask = self._reorder_state(self.encoder_mask, indices)
        self.encoder_key = self._reorder_state(self.encoder_key, indices)
        self.encoder_value = self._reorder_state(self.encoder_value, indices)
        self.decoder_prev_key = self._reorder_state(self.decoder_prev_key, indices)
        self.decoder_prev_value = self._reorder_state(self.decoder_prev_value, indices)
        return self


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def decode(self, *args, **kwargs) -> Tuple[torch.Tensor, Past]:
        """
        当模型进行解码时，使用这个函数。返回一个batch_size x vocab_size的结果与更新的Past状态。需要考虑一种特殊情况，即tokens长度不是1，即给定了
            解码句子开头的情况，这种情况需要查看Past中是否正确计算了decode的状态。

        :return: tensor:batch_size x vocab_size, past: Past
        """
        raise NotImplemented

    @abc.abstractmethod
    def reorder_past(self, indices: torch.LongTensor, past: Past):
        """
        根据indices中的index，将past的中状态置为正确的顺序。inplace改变

        :param torch.LongTensor indices:
        :param Past past:
        :return:
        """
        raise NotImplemented
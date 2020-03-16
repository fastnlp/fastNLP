from torch import nn
import torch
from fastNLP.modules import LSTM
from fastNLP import seq_len_to_mask
from torch.nn import TransformerEncoder
from typing import Union, Tuple
import numpy as np


class TransformerSeq2SeqEncoder(nn.Module):
    def __init__(self, embed: Union[Tuple[int, int], nn.Module, torch.Tensor, np.ndarray], num_layers: int = 6,
                 d_model: int = 512, n_head: int = 8, dim_ff: int = 2048, dropout: float = 0.1):
        super(TransformerSeq2SeqEncoder, self).__init__()
        self.embed = embed
        self.transformer = TransformerEncoder(nn.TransformerEncoderLayer(d_model, n_head), num_layers)

    def forward(self, words, seq_len):
        """

        :param words: batch, seq_len
        :param seq_len:
        :return: output: (batch, seq_len,dim) ; encoder_mask
        """
        words = self.embed(words)  # batch, seq_len, dim
        words = words.transpose(0, 1)
        encoder_mask = seq_len_to_mask(seq_len)  # batch, seq
        words = self.transformer(words, src_key_padding_mask=~encoder_mask)  # seq_len,batch,dim

        return words.transpose(0, 1), encoder_mask

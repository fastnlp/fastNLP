from fastNLP.modules.encoder.seq2seq_encoder import TransformerSeq2SeqEncoder
from fastNLP.modules.decoder.seq2seq_decoder import TransformerSeq2SeqDecoder, TransformerPast
from fastNLP.modules.decoder.seq2seq_generator import SequenceGenerator
import torch.nn as nn
import torch
from typing import Union, Tuple
import numpy as np


class TransformerSeq2SeqModel(nn.Module):
    def __init__(self, src_embed: Union[Tuple[int, int], nn.Module, torch.Tensor, np.ndarray],
                 tgt_embed: Union[Tuple[int, int], nn.Module, torch.Tensor, np.ndarray],
                 num_layers: int = 6,
                 d_model: int = 512, n_head: int = 8, dim_ff: int = 2048, dropout: float = 0.1,
                 output_embed: Union[Tuple[int, int], int, nn.Module, torch.Tensor, np.ndarray] = None,
                 bind_input_output_embed=False,
                 sos_id=None, eos_id=None):
        super().__init__()
        self.encoder = TransformerSeq2SeqEncoder(src_embed, num_layers, d_model, n_head, dim_ff, dropout)
        self.decoder = TransformerSeq2SeqDecoder(tgt_embed, num_layers, d_model, n_head, dim_ff, dropout, output_embed,
                                                 bind_input_output_embed)
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.num_layers = num_layers

    def forward(self, words, target, seq_len):  # todo:这里的target有sos和eos吗，参考一下lstm怎么写的
        encoder_output, encoder_mask = self.encoder(words, seq_len)
        past = TransformerPast(encoder_output, encoder_mask, self.num_layers)
        outputs = self.decoder(target, past, return_attention=False)

        return outputs

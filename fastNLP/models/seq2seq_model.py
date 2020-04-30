import torch.nn as nn
import torch
from typing import Union, Tuple
import numpy as np
from fastNLP.modules import TransformerSeq2SeqDecoder, TransformerSeq2SeqEncoder, TransformerPast


class TransformerSeq2SeqModel(nn.Module):  # todo 参考fairseq的FairseqModel的写法
    def __init__(self, src_embed: Union[Tuple[int, int], nn.Module, torch.Tensor, np.ndarray],
                 tgt_embed: Union[Tuple[int, int], nn.Module, torch.Tensor, np.ndarray],
                 num_layers: int = 6, d_model: int = 512, n_head: int = 8, dim_ff: int = 2048, dropout: float = 0.1,
                 output_embed: Union[Tuple[int, int], int, nn.Module, torch.Tensor, np.ndarray] = None,
                 bind_input_output_embed=False):
        super().__init__()
        self.encoder = TransformerSeq2SeqEncoder(src_embed, num_layers, d_model, n_head, dim_ff, dropout)
        self.decoder = TransformerSeq2SeqDecoder(tgt_embed, num_layers, d_model, n_head, dim_ff, dropout, output_embed,
                                                 bind_input_output_embed)

        self.num_layers = num_layers

    def forward(self, words, target, seq_len):
        encoder_output, encoder_mask = self.encoder(words, seq_len)
        past = TransformerPast(encoder_output, encoder_mask, self.num_layers)
        outputs = self.decoder(target, past, return_attention=False)

        return outputs

#!/usr/bin/python
# -*- coding: utf-8 -*-

# __author__="Danqing Wang"

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import torch
import torch.nn as nn

from .Encoder import Encoder
from tools.PositionEmbedding import get_sinusoid_encoding_table

from fastNLP.core.const import Const
from fastNLP.modules.encoder.transformer import TransformerEncoder

class TransformerModel(nn.Module):
    def __init__(self, hps, vocab):
        """
        
        :param hps: 
                min_kernel_size: min kernel size for cnn encoder
                max_kernel_size: max kernel size for cnn encoder
                output_channel: output_channel number for cnn encoder
                hidden_size: hidden size for transformer 
                n_layers: transfromer encoder layer
                n_head: multi head attention for transformer
                ffn_inner_hidden_size: FFN hiddens size
                atten_dropout_prob: dropout size
                doc_max_timesteps: max sentence number of the document
        :param vocab: 
        """
        super(TransformerModel, self).__init__()

        self._hps = hps
        self._vocab = vocab

        self.encoder = Encoder(hps, vocab)

        self.sent_embedding_size = (hps.max_kernel_size - hps.min_kernel_size + 1) * hps.output_channel
        self.hidden_size = hps.hidden_size

        self.n_head = hps.n_head
        self.d_v = self.d_k = int(self.hidden_size / self.n_head)
        self.d_inner = hps.ffn_inner_hidden_size
        self.num_layers = hps.n_layers

        self.projection = nn.Linear(self.sent_embedding_size, self.hidden_size)
        self.sent_pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(hps.doc_max_timesteps + 1, self.hidden_size, padding_idx=0), freeze=True)

        self.layer_stack = nn.ModuleList([
            TransformerEncoder.SubLayer(model_size=self.hidden_size, inner_size=self.d_inner, key_size=self.d_k, value_size=self.d_v,num_head=self.n_head, dropout=hps.atten_dropout_prob)
            for _ in range(self.num_layers)])

        self.wh = nn.Linear(self.hidden_size, 2)


    def forward(self, words, seq_len):
        """
        
        :param input: [batch_size, N, seq_len]
        :param input_len: [batch_size, N]
        :param return_atten: bool
        :return: 
        """
        # Sentence Encoder

        input = words
        input_len = seq_len

        self.sent_embedding = self.encoder(input) # [batch, N, Co * kernel_sizes]

        input_len = input_len.float()   # [batch, N]

        # -- Prepare masks
        batch_size, N = input_len.size()
        self.slf_attn_mask = input_len.eq(0.0)   # [batch, N]
        self.slf_attn_mask = self.slf_attn_mask.unsqueeze(1).expand(-1, N, -1)  # [batch, N, N]
        self.non_pad_mask = input_len.unsqueeze(-1)  # [batch, N, 1]

        input_doc_len = input_len.sum(dim=1).int()  # [batch]
        sent_pos = torch.Tensor([np.hstack((np.arange(1, doclen + 1), np.zeros(N - doclen))) for doclen in input_doc_len])
        sent_pos = sent_pos.long().cuda() if self._hps.cuda else sent_pos.long()

        enc_output_state = self.projection(self.sent_embedding)
        enc_input = enc_output_state + self.sent_pos_embed(sent_pos)

        # self.enc_slf_attn = self.enc_slf_attn * self.non_pad_mask
        enc_input_list = []
        for enc_layer in self.layer_stack:
            # enc_output = [batch_size, N, hidden_size = n_head * d_v]
            # enc_slf_attn = [n_head * batch_size, N, N]
            enc_input = enc_layer(enc_input, seq_mask=self.non_pad_mask, atte_mask_out=self.slf_attn_mask)
            enc_input_list += [enc_input]

        self.dec_output_state = torch.cat(enc_input_list[-4:])   # [4, batch_size, N, hidden_state]
        self.dec_output_state = self.dec_output_state.view(4, batch_size, N, -1)
        self.dec_output_state = self.dec_output_state.sum(0)

        p_sent = self.wh(self.dec_output_state) # [batch, N, 2]

        idx = None
        if self._hps.m == 0:
            prediction = p_sent.view(-1, 2).max(1)[1]
            prediction = prediction.view(batch_size, -1)
        else:
            mask_output = torch.exp(p_sent[:, :, 1])  # # [batch, N]
            mask_output = mask_output * input_len.float()
            topk, idx = torch.topk(mask_output, self._hps.m)
            prediction = torch.zeros(batch_size, N).scatter_(1, idx.data.cpu(), 1)
            prediction = prediction.long().view(batch_size, -1)

        if self._hps.cuda:
            prediction = prediction.cuda()

        # print((p_sent.size(), prediction.size(), idx.size()))
        # [batch, N, 2], [batch, N], [batch, hps.m]
        return {"pred": p_sent, "prediction": prediction, "pred_idx": idx}


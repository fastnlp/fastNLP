from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import *
from torch.distributions import *

from .Encoder import Encoder
from .DeepLSTM import DeepLSTM

from transformer.SubLayers import MultiHeadAttention,PositionwiseFeedForward

class SummarizationModel(nn.Module):
    def __init__(self, hps, embed):
        """
        
        :param hps: hyperparameters for the model
        :param embed: word embedding
        """
        super(SummarizationModel, self).__init__()

        self._hps = hps
        self.Train = (hps.mode == 'train')

        # sentence encoder
        self.encoder = Encoder(hps, embed)

        # Multi-layer highway lstm
        self.num_layers = hps.n_layers
        self.sent_embedding_size = (hps.max_kernel_size - hps.min_kernel_size + 1) * hps.output_channel
        self.lstm_hidden_size = hps.lstm_hidden_size
        self.recurrent_dropout = hps.recurrent_dropout_prob

        self.deep_lstm = DeepLSTM(self.sent_embedding_size, self.lstm_hidden_size, self.num_layers, self.recurrent_dropout,
                                                hps.use_orthnormal_init, hps.fix_mask, hps.cuda)

        # Multi-head attention
        self.n_head = hps.n_head
        self.d_v = self.d_k = int(self.lstm_hidden_size / hps.n_head)
        self.d_inner = hps.ffn_inner_hidden_size
        self.slf_attn = MultiHeadAttention(hps.n_head, self.lstm_hidden_size , self.d_k, self.d_v, dropout=hps.atten_dropout_prob)
        self.pos_ffn = PositionwiseFeedForward(self.d_v, self.d_inner, dropout = hps.ffn_dropout_prob)

        self.wh = nn.Linear(self.d_v, 2)


    def forward(self, words, seq_len):
        """
        
        :param input: [batch_size, N, seq_len], word idx long tensor 
        :param input_len: [batch_size, N], 1 for sentence and 0 for padding
        :return: 
            p_sent: [batch_size, N, 2]
            output_slf_attn: (option) [n_head, batch_size, N, N]
        """

        input = words
        input_len = seq_len

        # -- Sentence Encoder
        self.sent_embedding = self.encoder(input) # [batch, N, Co * kernel_sizes]

        # -- Multi-layer highway lstm
        input_len = input_len.float()   # [batch, N]
        self.inputs = [None] * (self.num_layers + 1)
        self.input_masks = [None] * (self.num_layers + 1)
        self.inputs[0] = self.sent_embedding.permute(1, 0, 2) # [N, batch, Co * kernel_sizes]
        self.input_masks[0] = input_len.permute(1, 0).unsqueeze(2)

        self.lstm_output_state = self.deep_lstm(self.inputs, self.input_masks, Train=self.train)    # [batch, N, hidden_size]

        # -- Prepare masks
        batch_size, N = input_len.size()
        slf_attn_mask = input_len.eq(0.0)   # [batch, N]， 1 for padding
        slf_attn_mask = slf_attn_mask.unsqueeze(1).expand(-1, N, -1)  # [batch, N, N]

        # -- Multi-head attention
        self.atten_output, self.output_slf_attn = self.slf_attn(self.lstm_output_state, self.lstm_output_state, self.lstm_output_state, mask=slf_attn_mask)
        self.atten_output *= input_len.unsqueeze(2)   # [batch_size, N, lstm_hidden_size = (n_head * d_v)]
        self.multi_atten_output = self.atten_output.view(batch_size, N, self.n_head, self.d_v)    # [batch_size, N, n_head, d_v]
        self.multi_atten_context = self.multi_atten_output[:, :, 0::2, :].sum(2) - self.multi_atten_output[:, :, 1::2, :].sum(2)   # [batch_size, N, d_v]

        # -- Position-wise Feed-Forward Networks
        self.output_state = self.pos_ffn(self.multi_atten_context)
        self.output_state = self.output_state * input_len.unsqueeze(2)  # [batch_size, N, d_v]

        p_sent = self.wh(self.output_state) # [batch, N, 2]

        idx = None
        if self._hps.m == 0:
            prediction = p_sent.view(-1, 2).max(1)[1]
            prediction = prediction.view(batch_size, -1)
        else:
            mask_output = torch.exp(p_sent[:, :, 1])  # # [batch, N]
            mask_output = mask_output.masked_fill(input_len.eq(0), 0)
            topk, idx = torch.topk(mask_output, self._hps.m)
            prediction = torch.zeros(batch_size, N).scatter_(1, idx.data.cpu(), 1)
            prediction = prediction.long().view(batch_size, -1)

        if self._hps.cuda:
            prediction = prediction.cuda()

        return {"p_sent": p_sent, "prediction": prediction, "pred_idx": idx}

r"""undocumented"""

__all__ = [
    "MultiHeadAttention",
    "BiAttention",
    "SelfAttention",
]

import math

import torch
import torch.nn.functional as F
from torch import nn

from .utils import initial_parameter
from .decoder.seq2seq_state import TransformerState


class DotAttention(nn.Module):
    r"""
    Transformer当中的DotAttention
    """

    def __init__(self, key_size, value_size, dropout=0.0):
        super(DotAttention, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.scale = math.sqrt(key_size)
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask_out=None):
        r"""

        :param Q: [..., seq_len_q, key_size]
        :param K: [..., seq_len_k, key_size]
        :param V: [..., seq_len_k, value_size]
        :param mask_out: [..., 1, seq_len] or [..., seq_len_q, seq_len_k]
        """
        output = torch.matmul(Q, K.transpose(-1, -2)) / self.scale
        if mask_out is not None:
            output.masked_fill_(mask_out, -1e9)
        output = self.softmax(output)
        output = self.drop(output)
        return torch.matmul(output, V)


class MultiHeadAttention(nn.Module):
    """
    Attention is all you need中提到的多头注意力

    """
    def __init__(self, d_model: int = 512, n_head: int = 8, dropout: float = 0.0, layer_idx: int = None):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = d_model // n_head
        self.layer_idx = layer_idx
        assert d_model % n_head == 0, "d_model should be divisible by n_head"
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.reset_parameters()

    def forward(self, query, key, value, key_mask=None, attn_mask=None, state=None):
        """

        :param query: batch x seq x dim
        :param key: batch x seq x dim
        :param value: batch x seq x dim
        :param key_mask: batch x seq 用于指示哪些key不要attend到；注意到mask为1的地方是要attend到的
        :param attn_mask: seq x seq, 用于mask掉attention map。 主要是用在训练时decoder端的self attention，下三角为1
        :param state: 过去的信息，在inference的时候会用到，比如encoder output、decoder的prev kv。这样可以减少计算。
        :return:
        """
        assert key.size() == value.size()
        if state is not None:
            assert self.layer_idx is not None
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()

        q = self.q_proj(query)  # batch x seq x dim
        q *= self.scaling
        k = v = None
        prev_k = prev_v = None

        # 从state中取kv
        if isinstance(state, TransformerState):  # 说明此时在inference阶段
            if qkv_same:  # 此时在decoder self attention
                prev_k = state.decoder_prev_key[self.layer_idx]
                prev_v = state.decoder_prev_value[self.layer_idx]
            else:  # 此时在decoder-encoder attention，直接将保存下来的key装载起来即可
                k = state.encoder_key[self.layer_idx]
                v = state.encoder_value[self.layer_idx]

        if k is None:
            k = self.k_proj(key)
            v = self.v_proj(value)

        if prev_k is not None:
            k = torch.cat((prev_k, k), dim=1)
            v = torch.cat((prev_v, v), dim=1)

        # 更新state
        if isinstance(state, TransformerState):
            if qkv_same:
                state.decoder_prev_key[self.layer_idx] = k
                state.decoder_prev_value[self.layer_idx] = v
            else:
                state.encoder_key[self.layer_idx] = k
                state.encoder_value[self.layer_idx] = v

        # 开始计算attention
        batch_size, q_len, d_model = query.size()
        k_len, v_len = k.size(1), v.size(1)
        q = q.reshape(batch_size, q_len, self.n_head, self.head_dim)
        k = k.reshape(batch_size, k_len, self.n_head, self.head_dim)
        v = v.reshape(batch_size, v_len, self.n_head, self.head_dim)

        attn_weights = torch.einsum('bqnh,bknh->bqkn', q, k)  # bs,q_len,k_len,n_head
        if key_mask is not None:
            _key_mask = ~key_mask[:, None, :, None].bool()  # batch,1,k_len,1
            attn_weights = attn_weights.masked_fill(_key_mask, -float('inf'))

        if attn_mask is not None:
            _attn_mask = attn_mask[None, :, :, None].eq(0)  # 1,q_len,k_len,n_head
            attn_weights = attn_weights.masked_fill(_attn_mask, -float('inf'))

        attn_weights = F.softmax(attn_weights, dim=2)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        output = torch.einsum('bqkn,bknh->bqnh', attn_weights, v)  # batch,q_len,n_head,head_dim
        output = output.reshape(batch_size, q_len, -1)
        output = self.out_proj(output)  # batch,q_len,dim

        return output, attn_weights

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def set_layer_idx(self, layer_idx):
        self.layer_idx = layer_idx


class AttentionLayer(nn.Module):
    def __init__(selfu, input_size, key_dim, value_dim, bias=False):
        """
        可用于LSTM2LSTM的序列到序列模型的decode过程中，该attention是在decode过程中根据上一个step的hidden计算对encoder结果的attention

        :param int input_size: 输入的大小
        :param int key_dim: 一般就是encoder_output输出的维度
        :param int value_dim: 输出的大小维度, 一般就是decoder hidden的大小
        :param bias:
        """
        super().__init__()

        selfu.input_proj = nn.Linear(input_size, key_dim, bias=bias)
        selfu.output_proj = nn.Linear(input_size + key_dim, value_dim, bias=bias)

    def forward(self, input, encode_outputs, encode_mask):
        """

        :param input: batch_size x input_size
        :param encode_outputs: batch_size x max_len x key_dim
        :param encode_mask: batch_size x max_len, 为0的地方为padding
        :return: hidden: batch_size x value_dim, scores: batch_size x max_len, normalized过的
        """

        # x: bsz x encode_hidden_size
        x = self.input_proj(input)

        # compute attention
        attn_scores = torch.matmul(encode_outputs, x.unsqueeze(-1)).squeeze(-1)  # b x max_len

        # don't attend over padding
        if encode_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encode_mask.eq(0),
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back

        attn_scores = F.softmax(attn_scores, dim=-1)  # srclen x bsz

        # sum weighted sources
        x = torch.matmul(attn_scores.unsqueeze(1), encode_outputs).squeeze(1)  # b x encode_hidden_size

        x = torch.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores


def _masked_softmax(tensor, mask):
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])
    result = F.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)
    return result.view(*tensor_shape)


def _weighted_sum(tensor, weights, mask):
    w_sum = weights.bmm(tensor)
    while mask.dim() < w_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(w_sum).contiguous().float()
    return w_sum * mask


class BiAttention(nn.Module):
    r"""
    Bi Attention module

    对于给定的两个向量序列 :math:`a_i` 和 :math:`b_j` , BiAttention模块将通过以下的公式来计算attention结果

    .. math::

        \begin{array}{ll} \\
            e_{ij} = {a}^{\mathrm{T}}_{i}{b}_{j} \\
            {\hat{a}}_{i} = \sum_{j=1}^{\mathcal{l}_{b}}{\frac{\mathrm{exp}(e_{ij})}{\sum_{k=1}^{\mathcal{l}_{b}}{\mathrm{exp}(e_{ik})}}}{b}_{j} \\
            {\hat{b}}_{j} = \sum_{i=1}^{\mathcal{l}_{a}}{\frac{\mathrm{exp}(e_{ij})}{\sum_{k=1}^{\mathcal{l}_{a}}{\mathrm{exp}(e_{ik})}}}{a}_{i} \\
        \end{array}

    """

    def forward(self, premise_batch, premise_mask, hypothesis_batch, hypothesis_mask):
        r"""
        :param torch.Tensor premise_batch: [batch_size, a_seq_len, hidden_size]
        :param torch.Tensor premise_mask: [batch_size, a_seq_len]
        :param torch.Tensor hypothesis_batch: [batch_size, b_seq_len, hidden_size]
        :param torch.Tensor hypothesis_mask: [batch_size, b_seq_len]
        :return: torch.Tensor attended_premises: [batch_size, a_seq_len, hidden_size] torch.Tensor attended_hypotheses: [batch_size, b_seq_len, hidden_size]
        """
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1)
                                              .contiguous())

        prem_hyp_attn = _masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = _masked_softmax(similarity_matrix.transpose(1, 2)
                                        .contiguous(),
                                        premise_mask)

        attended_premises = _weighted_sum(hypothesis_batch,
                                          prem_hyp_attn,
                                          premise_mask)
        attended_hypotheses = _weighted_sum(premise_batch,
                                            hyp_prem_attn,
                                            hypothesis_mask)

        return attended_premises, attended_hypotheses


class SelfAttention(nn.Module):
    r"""
    这是一个基于论文 `A structured self-attentive sentence embedding <https://arxiv.org/pdf/1703.03130.pdf>`_
    的Self Attention Module.
    """

    def __init__(self, input_size, attention_unit=300, attention_hops=10, drop=0.5, initial_method=None, ):
        r"""
        
        :param int input_size: 输入tensor的hidden维度
        :param int attention_unit: 输出tensor的hidden维度
        :param int attention_hops:
        :param float drop: dropout概率，默认值为0.5
        :param str initial_method: 初始化参数方法
        """
        super(SelfAttention, self).__init__()

        self.attention_hops = attention_hops
        self.ws1 = nn.Linear(input_size, attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, attention_hops, bias=False)
        self.I = torch.eye(attention_hops, requires_grad=False)
        self.I_origin = self.I
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        initial_parameter(self, initial_method)

    def _penalization(self, attention):
        r"""
        compute the penalization term for attention module
        """
        baz = attention.size(0)
        size = self.I.size()
        if len(size) != 3 or size[0] != baz:
            self.I = self.I_origin.expand(baz, -1, -1)
            self.I = self.I.to(device=attention.device)
        attention_t = torch.transpose(attention, 1, 2).contiguous()
        mat = torch.bmm(attention, attention_t) - self.I[:attention.size(0)]
        ret = (torch.sum(torch.sum((mat ** 2), 2), 1).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]

    def forward(self, input, input_origin):
        r"""
        :param torch.Tensor input: [batch_size, seq_len, hidden_size] 要做attention的矩阵
        :param torch.Tensor input_origin: [batch_size, seq_len] 原始token的index组成的矩阵，含有pad部分内容
        :return torch.Tensor output1: [batch_size, multi-head, hidden_size] 经过attention操作后输入矩阵的结果
        :return torch.Tensor output2: [1] attention惩罚项，是一个标量
        """
        input = input.contiguous()
        size = input.size()  # [bsz, len, nhid]

        input_origin = input_origin.expand(self.attention_hops, -1, -1)  # [hops,baz, len]
        input_origin = input_origin.transpose(0, 1).contiguous()  # [baz, hops,len]

        y1 = self.tanh(self.ws1(self.drop(input)))  # [baz,len,dim] -->[bsz,len, attention-unit]
        attention = self.ws2(y1).transpose(1, 2).contiguous()
        # [bsz,len, attention-unit]--> [bsz, len, hop]--> [baz,hop,len]

        attention = attention + (-999999 * (input_origin == 0).float())  # remove the weight on padding token.
        attention = F.softmax(attention, 2)  # [baz ,hop, len]
        return torch.bmm(attention, input), self._penalization(attention)  # output1 --> [baz ,hop ,nhid]

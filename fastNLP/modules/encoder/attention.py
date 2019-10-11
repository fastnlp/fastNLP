"""undocumented"""

__all__ = [
    "MultiHeadAttention",
    "BiAttention",
    "SelfAttention",
]

import math

import torch
import torch.nn.functional as F
from torch import nn

from fastNLP.modules.utils import initial_parameter


class DotAttention(nn.Module):
    """
    Transformer当中的DotAttention
    """

    def __init__(self, key_size, value_size, dropout=0.0):
        super(DotAttention, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.scale = math.sqrt(key_size)
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, Q, K, V, mask_out=None):
        """

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
    Transformer当中的MultiHeadAttention
    """

    def __init__(self, input_size, key_size, value_size, num_head, dropout=0.1):
        """
        
        :param input_size: int, 输入维度的大小。同时也是输出维度的大小。
        :param key_size: int, 每个head的维度大小。
        :param value_size: int，每个head中value的维度。
        :param num_head: int，head的数量。
        :param dropout: float。
        """
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.key_size = key_size
        self.value_size = value_size
        self.num_head = num_head

        in_size = key_size * num_head
        self.q_in = nn.Linear(input_size, in_size)
        self.k_in = nn.Linear(input_size, in_size)
        self.v_in = nn.Linear(input_size, in_size)
        self.attention = DotAttention(key_size=key_size, value_size=value_size, dropout=dropout)
        self.out = nn.Linear(value_size * num_head, input_size)
        self.reset_parameters()

    def reset_parameters(self):
        sqrt = math.sqrt
        nn.init.normal_(self.q_in.weight, mean=0, std=sqrt(1.0 / self.input_size))
        nn.init.normal_(self.k_in.weight, mean=0, std=sqrt(1.0 / self.input_size))
        nn.init.normal_(self.v_in.weight, mean=0, std=sqrt(1.0 / self.input_size))
        nn.init.normal_(self.out.weight, mean=0, std=sqrt(1.0 / self.input_size))

    def forward(self, Q, K, V, atte_mask_out=None):
        """

        :param Q: [batch, seq_len_q, model_size]
        :param K: [batch, seq_len_k, model_size]
        :param V: [batch, seq_len_k, model_size]
        :param seq_mask: [batch, seq_len]
        """
        batch, sq, _ = Q.size()
        sk = K.size(1)
        d_k, d_v, n_head = self.key_size, self.value_size, self.num_head
        # input linear
        q = self.q_in(Q).view(batch, sq, n_head, d_k).transpose(1, 2)
        k = self.k_in(K).view(batch, sk, n_head, d_k).transpose(1, 2)
        v = self.v_in(V).view(batch, sk, n_head, d_v).transpose(1, 2)

        if atte_mask_out is not None:
            atte_mask_out = atte_mask_out[:,None,:,:] # [bsz,1,1,len]
        atte = self.attention(q, k, v, atte_mask_out).view(batch, n_head, sq, d_v)

        # concat all heads, do output linear
        atte = atte.transpose(1, 2).contiguous().view(batch, sq, -1)
        output = self.out(atte)
        return output


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
        """
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
    """
    这是一个基于论文 `A structured self-attentive sentence embedding <https://arxiv.org/pdf/1703.03130.pdf>`_
    的Self Attention Module.
    """

    def __init__(self, input_size, attention_unit=300, attention_hops=10, drop=0.5, initial_method=None, ):
        """
        
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
        """
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
        """
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

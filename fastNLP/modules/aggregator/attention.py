__all__ = [
    "MultiHeadAttention"
]

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..dropout import TimestepDropout

from ..utils import initial_parameter


class DotAttention(nn.Module):
    """
    .. todo::
        补上文档
    """
    
    def __init__(self, key_size, value_size, dropout=0):
        super(DotAttention, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.scale = math.sqrt(key_size)
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)
    
    def forward(self, Q, K, V, mask_out=None):
        """

        :param Q: [batch, seq_len_q, key_size]
        :param K: [batch, seq_len_k, key_size]
        :param V: [batch, seq_len_k, value_size]
        :param mask_out: [batch, 1, seq_len] or [batch, seq_len_q, seq_len_k]
        """
        output = torch.matmul(Q, K.transpose(1, 2)) / self.scale
        if mask_out is not None:
            output.masked_fill_(mask_out, -1e8)
        output = self.softmax(output)
        output = self.drop(output)
        return torch.matmul(output, V)


class MultiHeadAttention(nn.Module):
    """
    别名：:class:`fastNLP.modules.MultiHeadAttention`   :class:`fastNLP.modules.aggregator.attention.MultiHeadAttention`


    :param input_size: int, 输入维度的大小。同时也是输出维度的大小。
    :param key_size: int, 每个head的维度大小。
    :param value_size: int，每个head中value的维度。
    :param num_head: int，head的数量。
    :param dropout: float。
    """
    
    def __init__(self, input_size, key_size, value_size, num_head, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.key_size = key_size
        self.value_size = value_size
        self.num_head = num_head
        
        in_size = key_size * num_head
        self.q_in = nn.Linear(input_size, in_size)
        self.k_in = nn.Linear(input_size, in_size)
        self.v_in = nn.Linear(input_size, in_size)
        # follow the paper, do not apply dropout within dot-product
        self.attention = DotAttention(key_size=key_size, value_size=value_size, dropout=0)
        self.out = nn.Linear(value_size * num_head, input_size)
        self.drop = TimestepDropout(dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        sqrt = math.sqrt
        nn.init.normal_(self.q_in.weight, mean=0, std=sqrt(2.0 / (self.input_size + self.key_size)))
        nn.init.normal_(self.k_in.weight, mean=0, std=sqrt(2.0 / (self.input_size + self.key_size)))
        nn.init.normal_(self.v_in.weight, mean=0, std=sqrt(2.0 / (self.input_size + self.value_size)))
        nn.init.xavier_normal_(self.out.weight)
    
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
        q = self.q_in(Q).view(batch, sq, n_head, d_k)
        k = self.k_in(K).view(batch, sk, n_head, d_k)
        v = self.v_in(V).view(batch, sk, n_head, d_v)
        
        # transpose q, k and v to do batch attention
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, sq, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, sk, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, sk, d_v)
        if atte_mask_out is not None:
            atte_mask_out = atte_mask_out.repeat(n_head, 1, 1)
        atte = self.attention(q, k, v, atte_mask_out).view(n_head, batch, sq, d_v)
        
        # concat all heads, do output linear
        atte = atte.permute(1, 2, 0, 3).contiguous().view(batch, sq, -1)
        output = self.drop(self.out(atte))
        return output


class BiAttention(nn.Module):
    r"""Bi Attention module
    
    .. todo::
        这个模块的负责人来继续完善一下
        
    Calculate Bi Attention matrix `e`
    
    .. math::
    
        \begin{array}{ll} \\
            e_ij = {a}^{\mathbf{T}}_{i}{b}_{j} \\
            a_i =
            b_j =
        \end{array}
        
    """
    
    def __init__(self):
        super(BiAttention, self).__init__()
        self.inf = 10e12
    
    def forward(self, in_x1, in_x2, x1_len, x2_len):
        """
        :param torch.Tensor in_x1: [batch_size, x1_seq_len, hidden_size] 第一句的特征表示
        :param torch.Tensor in_x2: [batch_size, x2_seq_len, hidden_size] 第二句的特征表示
        :param torch.Tensor x1_len: [batch_size, x1_seq_len] 第一句的0/1mask矩阵
        :param torch.Tensor x2_len: [batch_size, x2_seq_len] 第二句的0/1mask矩阵
        :return: torch.Tensor out_x1: [batch_size, x1_seq_len, hidden_size] 第一句attend到的特征表示
            torch.Tensor out_x2: [batch_size, x2_seq_len, hidden_size] 第一句attend到的特征表示
        
        """
        
        assert in_x1.size()[0] == in_x2.size()[0]
        assert in_x1.size()[2] == in_x2.size()[2]
        # The batch size and hidden size must be equal.
        assert in_x1.size()[1] == x1_len.size()[1] and in_x2.size()[1] == x2_len.size()[1]
        # The seq len in in_x and x_len must be equal.
        assert in_x1.size()[0] == x1_len.size()[0] and x1_len.size()[0] == x2_len.size()[0]
        
        batch_size = in_x1.size()[0]
        x1_max_len = in_x1.size()[1]
        x2_max_len = in_x2.size()[1]
        
        in_x2_t = torch.transpose(in_x2, 1, 2)  # [batch_size, hidden_size, x2_seq_len]
        
        attention_matrix = torch.bmm(in_x1, in_x2_t)  # [batch_size, x1_seq_len, x2_seq_len]
        
        a_mask = x1_len.le(0.5).float() * -self.inf  # [batch_size, x1_seq_len]
        a_mask = a_mask.view(batch_size, x1_max_len, -1)
        a_mask = a_mask.expand(-1, -1, x2_max_len)  # [batch_size, x1_seq_len, x2_seq_len]
        b_mask = x2_len.le(0.5).float() * -self.inf
        b_mask = b_mask.view(batch_size, -1, x2_max_len)
        b_mask = b_mask.expand(-1, x1_max_len, -1)  # [batch_size, x1_seq_len, x2_seq_len]
        
        attention_a = F.softmax(attention_matrix + a_mask, dim=2)  # [batch_size, x1_seq_len, x2_seq_len]
        attention_b = F.softmax(attention_matrix + b_mask, dim=1)  # [batch_size, x1_seq_len, x2_seq_len]
        
        out_x1 = torch.bmm(attention_a, in_x2)  # [batch_size, x1_seq_len, hidden_size]
        attention_b_t = torch.transpose(attention_b, 1, 2)
        out_x2 = torch.bmm(attention_b_t, in_x1)  # [batch_size, x2_seq_len, hidden_size]
        
        return out_x1, out_x2


class SelfAttention(nn.Module):
    """
    Self Attention Module.
    
    :param int input_size: 输入tensor的hidden维度
    :param int attention_unit: 输出tensor的hidden维度
    :param int attention_hops:
    :param float drop: dropout概率，默认值为0.5
    :param str initial_method: 初始化参数方法
    """
    
    def __init__(self, input_size, attention_unit=300, attention_hops=10, drop=0.5, initial_method=None, ):
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
        :param torch.Tensor input: [baz, senLen, h_dim] 要做attention的矩阵
        :param torch.Tensor input_origin: [baz , senLen] 原始token的index组成的矩阵，含有pad部分内容
        :return torch.Tensor output1: [baz, multi-head , h_dim] 经过attention操作后输入矩阵的结果
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

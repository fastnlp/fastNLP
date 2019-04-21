import math

import torch
import torch.nn.functional as F
from torch import nn

from fastNLP.modules.dropout import TimestepDropout
from fastNLP.modules.utils import mask_softmax


class Attention(torch.nn.Module):
    def __init__(self, normalize=False):
        super(Attention, self).__init__()
        self.normalize = normalize

    def forward(self, query, memory, mask):
        similarities = self._atten_forward(query, memory)
        if self.normalize:
            return mask_softmax(similarities, mask)
        return similarities

    def _atten_forward(self, query, memory):
        raise NotImplementedError


class DotAttention(nn.Module):
    def __init__(self, key_size, value_size, dropout=0.1):
        super(DotAttention, self).__init__()
        self.key_size = key_size
        self.value_size = value_size
        self.scale = math.sqrt(key_size)
        self.drop = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, Q, K, V, mask_out=None):
        """

        :param Q: [batch, seq_len, key_size]
        :param K: [batch, seq_len, key_size]
        :param V: [batch, seq_len, value_size]
        :param mask_out: [batch, seq_len]
        """
        output = torch.matmul(Q, K.transpose(1, 2)) / self.scale
        if mask_out is not None:
            output.masked_fill_(mask_out, -float('inf'))
        output = self.softmax(output)
        output = self.drop(output)
        return torch.matmul(output, V)


class MultiHeadAttention(nn.Module):
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
        self.attention = DotAttention(key_size=key_size, value_size=value_size)
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

        :param Q: [batch, seq_len, model_size]
        :param K: [batch, seq_len, model_size]
        :param V: [batch, seq_len, model_size]
        :param seq_mask: [batch, seq_len]
        """
        batch, seq_len, _ = Q.size()
        d_k, d_v, n_head = self.key_size, self.value_size, self.num_head
        # input linear
        q = self.q_in(Q).view(batch, seq_len, n_head, d_k)
        k = self.k_in(K).view(batch, seq_len, n_head, d_k)
        v = self.v_in(V).view(batch, seq_len, n_head, d_k)

        # transpose q, k and v to do batch attention
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, seq_len, d_v)
        if atte_mask_out is not None:
            atte_mask_out = atte_mask_out.repeat(n_head, 1, 1)
        atte = self.attention(q, k, v, atte_mask_out).view(n_head, batch, seq_len, d_v)

        # concat all heads, do output linear
        atte = atte.permute(1, 2, 0, 3).contiguous().view(batch, seq_len, -1)
        output = self.drop(self.out(atte))
        return output


class BiAttention(nn.Module):
    """Bi Attention module
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

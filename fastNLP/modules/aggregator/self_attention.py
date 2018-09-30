import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from fastNLP.modules.utils import initial_parameter


class SelfAttention(nn.Module):
    """
    Self Attention Module.

    Args:
    input_size: int, the size for the input vector
    dim: int, the width of weight matrix.
    num_vec: int, the number of encoded vectors
    """

    def __init__(self, input_size, attention_unit=350, attention_hops=10, drop=0.5, initial_method=None,
                 use_cuda=False):
        super(SelfAttention, self).__init__()

        self.attention_hops = attention_hops
        self.ws1 = nn.Linear(input_size, attention_unit, bias=False)
        self.ws2 = nn.Linear(attention_unit, attention_hops, bias=False)
        if use_cuda:
            self.I = Variable(torch.eye(attention_hops).cuda(), requires_grad=False)
        else:
            self.I = Variable(torch.eye(attention_hops), requires_grad=False)
        self.I_origin = self.I
        self.drop = nn.Dropout(drop)
        self.tanh = nn.Tanh()
        initial_parameter(self, initial_method)

    def penalization(self, attention):
        """
        compute the penalization term for attention module
        """
        baz = attention.size(0)
        size = self.I.size()
        if len(size) != 3 or size[0] != baz:
            self.I = self.I_origin.expand(baz, -1, -1)
        attentionT = torch.transpose(attention, 1, 2).contiguous()
        mat = torch.bmm(attention, attentionT) - self.I[:attention.size(0)]
        ret = (torch.sum(torch.sum((mat ** 2), 2), 1).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]

    def forward(self, input, input_origin):
        """
        :param input:  the matrix to do attention.              [baz, senLen, h_dim]
        :param inp:  then token index include pad token( 0 )   [baz , senLen]
        :return output1: the input matrix after attention operation   [baz, multi-head , h_dim]
        :return output2: the attention penalty term, a scalar  [1]
        """
        input = input.contiguous()
        size = input.size()  # [bsz, len, nhid]

        input_origin = input_origin.expand(self.attention_hops, -1, -1)  # [hops,baz, len]
        input_origin = input_origin.transpose(0, 1).contiguous()  # [baz, hops,len]

        y1 = self.tanh(self.ws1(self.drop(input)))  # [baz,len,dim] -->[bsz,len, attention-unit]
        attention = self.ws2(y1).transpose(1,
                                           2).contiguous()  # [bsz,len, attention-unit]--> [bsz, len, hop]--> [baz,hop,len]

        attention = attention + (-999999 * (input_origin == 0).float())  # remove the weight on padding token.
        attention = F.softmax(attention, 2)  # [baz ,hop, len]
        return torch.bmm(attention, input), self.penalization(attention)  # output1 --> [baz ,hop ,nhid]

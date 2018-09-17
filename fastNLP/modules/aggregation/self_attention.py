import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


from fastNLP.modules.utils import initial_parameter
class SelfAttention(nn.Module):
    """
    Self Attention Module.

    Args:
    input_size: int, the size for the input vector
    dim: int, the width of weight matrix.
    num_vec: int, the number of encoded vectors
    """

    def __init__(self, input_size, dim=10, num_vec=10 ,drop = 0.5 ,initial_method =None):
        super(SelfAttention, self).__init__()
        # self.W_s1 = nn.Parameter(torch.randn(dim, input_size), requires_grad=True)
        # self.W_s2 = nn.Parameter(torch.randn(num_vec, dim), requires_grad=True)
        self.attention_hops = num_vec

        self.ws1 = nn.Linear(input_size, dim, bias=False)
        self.ws2 = nn.Linear(dim, num_vec, bias=False)
        self.drop = nn.Dropout(drop)
        self.softmax = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()
        initial_parameter(self, initial_method)
    def penalization(self, A):
        """
        compute the penalization term for attention module
        """
        if self.W_s1.is_cuda:
            I = Variable(torch.eye(A.size(1)).cuda(), requires_grad=False)
        else:
            I = Variable(torch.eye(A.size(1)), requires_grad=False)
        M = torch.matmul(A, torch.transpose(A, 1, 2)) - I
        M = M.view(M.size(0), -1)
        return torch.sum(M ** 2, dim=1)
        
    def forward(self, outp ,inp):
        #  the following code can not be use because some word are padding ,these is not such module!

        # inter = self.tanh(torch.matmul(self.W_s1, torch.transpose(x, 1, 2))) # []
        # A = self.softmax(torch.matmul(self.W_s2, inter))
        # out = torch.matmul(A, x)
        # out = out.view(out.size(0), -1)
        # penalty = self.penalization(A)
        # return out, penalty
        outp = outp.contiguous()
        size = outp.size()  # [bsz, len, nhid]

        compressed_embeddings = outp.view(-1, size[2])  # [bsz*len, nhid*2]
        transformed_inp = torch.transpose(inp, 0, 1).contiguous()  # [bsz, len]
        transformed_inp = transformed_inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

        hbar = self.tanh(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        attention = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        attention = torch.transpose(attention, 1, 2).contiguous()  # [bsz, hop, len]
        penalized_alphas = attention + (
            -10000 * (concatenated_inp == 0).float())
        # [bsz, hop, len] + [bsz, hop, len]
        attention = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        attention = attention.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(attention, outp), attention  # output --> [baz ,hop ,nhid]




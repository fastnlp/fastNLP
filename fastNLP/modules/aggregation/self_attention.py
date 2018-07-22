import torch
import torch.nn as nn
from torch.autograd import Variable


class SelfAttention(nn.Module):
    """
    Self Attention Module.

    Args:
    input_size: int, the size for the input vector
    dim: int, the width of weight matrix.
    num_vec: int, the number of encoded vectors
    """

    def __init__(self, input_size, dim=10, num_vec=10):
        super(SelfAttention, self).__init__()
        self.W_s1 = nn.Parameter(torch.randn(dim, input_size), requires_grad=True)
        self.W_s2 = nn.Parameter(torch.randn(num_vec, dim), requires_grad=True)
        self.softmax = nn.Softmax(dim=2)
        self.tanh = nn.Tanh()

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
        
    def forward(self, x):
        inter = self.tanh(torch.matmul(self.W_s1, torch.transpose(x, 1, 2)))
        A = self.softmax(torch.matmul(self.W_s2, inter))
        out = torch.matmul(A, x)
        out = out.view(out.size(0), -1)
        penalty = self.penalization(A)
        return out, penalty


import torch.nn as nn


class Embedding(nn.Module):
    """A simple lookup table.

    :param int nums: the size of the lookup table
    :param int dims: the size of each vector
    :param int padding_idx: pads the tensor with zeros whenever it encounters this index
    :param bool sparse: If True, gradient matrix will be a sparse tensor. In this case, only optim.SGD(cuda and cpu) and optim.Adagrad(cpu) can be used
    """
    def __init__(self, nums, dims, padding_idx=0, sparse=False, init_emb=None, dropout=0.0):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(nums, dims, padding_idx, sparse=sparse)
        if init_emb is not None:
            self.embed.weight = nn.Parameter(init_emb)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embed(x)
        return self.dropout(x)

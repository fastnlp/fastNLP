import torch
import torch.nn as nn

class Lookuptable(nn.Module):
    """
    A simple lookup table

    Args:
    nums : the size of the lookup table
    dims : the size of each vector
    padding_idx : pads the tensor with zeros whenever it encounters this index
    sparse : If True, gradient matrix will be a sparse tensor. In this case,
    only optim.SGD(cuda and cpu) and optim.Adagrad(cpu) can be used
    """
    def __init__(self, nums, dims, padding_idx=0, sparse=False):
        super(Lookuptable, self).__init__()
        self.embed = nn.Embedding(nums, dims, padding_idx, sparse=sparse)
        
    def forward(self, x):
        return self.embed(x)

if __name__ == "__main__":
    model = Lookuptable(10, 20)

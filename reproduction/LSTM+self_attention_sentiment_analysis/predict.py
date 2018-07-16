import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """
    A two layers perceptron for classification.

    Output : Unnormalized possibility distribution
    Args:
    input_size : the size of input
    hidden_size : the size of hidden layer
    output_size : the size of output
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP,self).__init__()
        self.L1 = nn.Linear(input_size, hidden_size)
        self.L2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.L2(F.relu(self.L1(x)))
        return out

if __name__ == "__main__":
    MLP(20, 30, 20)
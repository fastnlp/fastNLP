import torch

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

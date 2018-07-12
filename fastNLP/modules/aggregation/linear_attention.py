from fastNLP.modules.aggregation.attention import Attention


class LinearAttention(Attention):
    def __init__(self, normalize=False):
        super(LinearAttention, self).__init__(normalize)

    def _atten_forward(self, query, memory):
        raise NotImplementedError

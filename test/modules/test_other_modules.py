import unittest

import torch

from fastNLP.modules.encoder.star_transformer import StarTransformer


class TestStarTransformer(unittest.TestCase):
    def test_1(self):
        model = StarTransformer(num_layers=6, hidden_size=100, num_head=8, head_dim=20, max_len=100)
        x = torch.rand(16, 45, 100)
        mask = torch.ones(16, 45).byte()
        y, yn = model(x, mask)
        self.assertEqual(tuple(y.size()), (16, 45, 100))
        self.assertEqual(tuple(yn.size()), (16, 100))

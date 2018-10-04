import unittest

import torch

from fastNLP.modules.other_modules import GroupNorm, LayerNormalization, BiLinear, BiAffine


class TestGroupNorm(unittest.TestCase):
    def test_case_1(self):
        gn = GroupNorm(num_features=1, num_groups=10, eps=1.5e-5)
        x = torch.randn((20, 50, 10))
        y = gn(x)


class TestLayerNormalization(unittest.TestCase):
    def test_case_1(self):
        ln = LayerNormalization(d_hid=5, eps=2e-3)
        x = torch.randn((20, 50, 5))
        y = ln(x)


class TestBiLinear(unittest.TestCase):
    def test_case_1(self):
        bl = BiLinear(n_left=5, n_right=5, n_out=10, bias=True)
        x_left = torch.randn((7, 10, 20, 5))
        x_right = torch.randn((7, 10, 20, 5))
        y = bl(x_left, x_right)
        print(bl)
        bl2 = BiLinear(n_left=15, n_right=15, n_out=10, bias=True)


class TestBiAffine(unittest.TestCase):
    def test_case_1(self):
        batch_size = 16
        encoder_length = 21
        decoder_length = 32
        layer = BiAffine(10, 10, 25, biaffine=True)
        decoder_input = torch.randn((batch_size, encoder_length, 10))
        encoder_input = torch.randn((batch_size, decoder_length, 10))
        y = layer(decoder_input, encoder_input)
        self.assertEqual(tuple(y.shape), (batch_size, 25, encoder_length, decoder_length))

    def test_case_2(self):
        batch_size = 16
        encoder_length = 21
        decoder_length = 32
        layer = BiAffine(10, 10, 25, biaffine=False)
        decoder_input = torch.randn((batch_size, encoder_length, 10))
        encoder_input = torch.randn((batch_size, decoder_length, 10))
        y = layer(decoder_input, encoder_input)
        self.assertEqual(tuple(y.shape), (batch_size, 25, encoder_length, 1))

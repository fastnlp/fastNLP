import unittest

import torch

from fastNLP.modules.encoder.char_encoder import ConvolutionCharEncoder, LSTMCharEncoder


class TestCharEmbed(unittest.TestCase):
    def test_case_1(self):
        batch_size = 128
        char_emb = 100
        word_length = 1
        x = torch.Tensor(batch_size, char_emb, word_length)
        x = x.transpose(1, 2)

        cce = ConvolutionCharEncoder(char_emb)
        y = cce(x)
        self.assertEqual(tuple(x.shape), (batch_size, word_length, char_emb))
        print("CNN Char Emb input: ", x.shape)
        self.assertEqual(tuple(y.shape), (batch_size, char_emb, 1))
        print("CNN Char Emb output: ", y.shape)  # [128, 100]

        lce = LSTMCharEncoder(char_emb)
        o = lce(x)
        self.assertEqual(tuple(x.shape), (batch_size, word_length, char_emb))
        print("LSTM Char Emb input: ", x.shape)
        self.assertEqual(tuple(o.shape), (batch_size, char_emb, 1))
        print("LSTM Char Emb size: ", o.shape)

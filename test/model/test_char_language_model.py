import unittest

import numpy as np
import torch

from fastNLP.models.char_language_model import CharLM


class TestCharLM(unittest.TestCase):
    def test_case_1(self):
        char_emb_dim = 50
        word_emb_dim = 50
        vocab_size = 1000
        num_char = 24
        max_word_len = 21
        num_seq = 64
        seq_len = 32

        model = CharLM(char_emb_dim, word_emb_dim, vocab_size, num_char)

        x = torch.from_numpy(np.random.randint(0, num_char, size=(num_seq, seq_len, max_word_len + 2)))

        self.assertEqual(tuple(x.shape), (num_seq, seq_len, max_word_len + 2))
        y = model(x)
        self.assertEqual(tuple(y.shape), (num_seq * seq_len, vocab_size))

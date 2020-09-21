import unittest

import torch

from fastNLP.modules.encoder.seq2seq_encoder import TransformerSeq2SeqEncoder, LSTMSeq2SeqEncoder
from fastNLP import Vocabulary
from fastNLP.embeddings import StaticEmbedding


class TestTransformerSeq2SeqEncoder(unittest.TestCase):
    def test_case(self):
        vocab = Vocabulary().add_word_lst("This is a test .".split())
        embed = StaticEmbedding(vocab, embedding_dim=5)
        encoder = TransformerSeq2SeqEncoder(embed, num_layers=2, d_model=10, n_head=2)
        words_idx = torch.LongTensor([0, 1, 2]).unsqueeze(0)
        seq_len = torch.LongTensor([3])
        encoder_output, encoder_mask = encoder(words_idx, seq_len)
        self.assertEqual(encoder_output.size(), (1, 3, 10))


class TestBiLSTMEncoder(unittest.TestCase):
    def test_case(self):
        vocab = Vocabulary().add_word_lst("This is a test .".split())
        embed = StaticEmbedding(vocab, embedding_dim=5)
        encoder = LSTMSeq2SeqEncoder(embed, hidden_size=5, num_layers=1)
        words_idx = torch.LongTensor([0, 1, 2]).unsqueeze(0)
        seq_len = torch.LongTensor([3])

        encoder_output, encoder_mask = encoder(words_idx, seq_len)
        self.assertEqual(encoder_mask.size(), (1, 3))

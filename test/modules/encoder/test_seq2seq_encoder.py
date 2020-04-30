import unittest

import torch

from fastNLP.modules.encoder.seq2seq_encoder import TransformerSeq2SeqEncoder, BiLSTMEncoder
from fastNLP import Vocabulary
from fastNLP.embeddings import StaticEmbedding

class TestTransformerSeq2SeqEncoder(unittest.TestCase):
    def test_case(self):
        vocab = Vocabulary().add_word_lst("This is a test .".split())
        embed = StaticEmbedding(vocab, embedding_dim=512)
        encoder = TransformerSeq2SeqEncoder(embed)
        words_idx = torch.LongTensor([0, 1, 2]).unsqueeze(0)
        seq_len = torch.LongTensor([3])
        outputs, mask = encoder(words_idx, seq_len)

        print(outputs)
        print(mask)
        self.assertEqual(outputs.size(), (1, 3, 512))


class TestBiLSTMEncoder(unittest.TestCase):
    def test_case(self):
        vocab = Vocabulary().add_word_lst("This is a test .".split())
        embed = StaticEmbedding(vocab, embedding_dim=300)
        encoder = BiLSTMEncoder(embed, hidden_size=300)
        words_idx = torch.LongTensor([0, 1, 2]).unsqueeze(0)
        seq_len = torch.LongTensor([3])

        outputs, hx = encoder(words_idx, seq_len)

        # print(outputs)
        # print(hx)
        self.assertEqual(outputs.size(), (1, 3, 300))

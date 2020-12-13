import unittest

import torch

from fastNLP import Vocabulary, DataSet, Instance
from fastNLP.embeddings.char_embedding import LSTMCharEmbedding, CNNCharEmbedding


class TestCharEmbed(unittest.TestCase):
    def test_case_1(self):
        ds = DataSet([Instance(words=['hello', 'world']), Instance(words=['Jack'])])
        vocab = Vocabulary().from_dataset(ds, field_name='words')
        self.assertEqual(len(vocab), 5)
        embed = LSTMCharEmbedding(vocab, embed_size=60)
        x = torch.LongTensor([[2, 1, 0], [4, 3, 4]])
        y = embed(x)
        self.assertEqual(tuple(y.size()), (2, 3, 60))

    def test_case_2(self):
        ds = DataSet([Instance(words=['hello', 'world']), Instance(words=['Jack'])])
        vocab = Vocabulary().from_dataset(ds, field_name='words')
        self.assertEqual(len(vocab), 5)
        embed = CNNCharEmbedding(vocab, embed_size=60)
        x = torch.LongTensor([[2, 1, 0], [4, 3, 4]])
        y = embed(x)
        self.assertEqual(tuple(y.size()), (2, 3, 60))

import unittest

import torch

from fastNLP import Vocabulary, DataSet, Instance
from fastNLP.embeddings import LSTMCharEmbedding, CNNCharEmbedding, StackEmbedding


class TestCharEmbed(unittest.TestCase):
    def test_case_1(self):
        ds = DataSet([Instance(words=['hello', 'world']), Instance(words=['hello', 'Jack'])])
        vocab = Vocabulary().from_dataset(ds, field_name='words')
        self.assertEqual(len(vocab), 5)
        cnn_embed = CNNCharEmbedding(vocab, embed_size=60)
        lstm_embed = LSTMCharEmbedding(vocab, embed_size=70)
        embed = StackEmbedding([cnn_embed, lstm_embed])
        x = torch.LongTensor([[2, 1, 0], [4, 3, 4]])
        y = embed(x)
        self.assertEqual(tuple(y.size()), (2, 3, 130))

    def test_case_2(self):
        # 测试只需要拥有一样的index就可以concat
        ds = DataSet([Instance(words=['hello', 'world']), Instance(words=['hello', 'Jack'])])
        vocab1 = Vocabulary().from_dataset(ds, field_name='words')
        vocab2 = Vocabulary().from_dataset(ds, field_name='words')
        self.assertEqual(len(vocab1), 5)
        cnn_embed = CNNCharEmbedding(vocab1, embed_size=60)
        lstm_embed = LSTMCharEmbedding(vocab2, embed_size=70)
        embed = StackEmbedding([cnn_embed, lstm_embed])
        x = torch.LongTensor([[2, 1, 0], [4, 3, 4]])
        y = embed(x)
        self.assertEqual(tuple(y.size()), (2, 3, 130))


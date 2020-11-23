import unittest

import torch
import os

from fastNLP import DataSet, Vocabulary
from fastNLP.embeddings.transformers_embedding import TransformersEmbedding, TransformersWordPieceEncoder


class TransformersEmbeddingTest(unittest.TestCase):
    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_transformers_embedding_1(self):
        from transformers import ElectraModel, ElectraTokenizer
        weight_path = "google/electra-small-generator"
        vocab = Vocabulary().add_word_lst("this is a test . [SEP] NotInRoberta".split())
        model = ElectraModel.from_pretrained(weight_path)
        tokenizer = ElectraTokenizer.from_pretrained(weight_path)

        embed = TransformersEmbedding(vocab, model, tokenizer, word_dropout=0.1)

        words = torch.LongTensor([[2, 3, 4, 1]])
        result = embed(words)
        self.assertEqual(result.size(), (1, 4, model.config.hidden_size))


class TransformersWordPieceEncoderTest(unittest.TestCase):
    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_transformers_embedding_1(self):
        from transformers import ElectraModel, ElectraTokenizer
        weight_path = "google/electra-small-generator"
        model = ElectraModel.from_pretrained(weight_path)
        tokenizer = ElectraTokenizer.from_pretrained(weight_path)
        encoder = TransformersWordPieceEncoder(model, tokenizer)
        ds = DataSet({'words': ["this is a test . [SEP]".split()]})
        encoder.index_datasets(ds, field_name='words')
        self.assertTrue(ds.has_field('word_pieces'))
        result = encoder(torch.LongTensor([[1,2,3,4]]))
        self.assertEqual(result.size(), (1, 4, model.config.hidden_size))

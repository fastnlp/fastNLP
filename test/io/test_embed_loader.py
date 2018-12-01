import unittest

from fastNLP.core.vocabulary import Vocabulary
from fastNLP.io.embed_loader import EmbedLoader


class TestEmbedLoader(unittest.TestCase):
    def test_case(self):
        vocab = Vocabulary()
        vocab.update(["the", "in", "I", "to", "of", "hahaha"])
        embedding = EmbedLoader().fast_load_embedding(50, "../data_for_tests/glove.6B.50d_test.txt", vocab)
        self.assertEqual(tuple(embedding.shape), (len(vocab), 50))

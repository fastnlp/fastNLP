import unittest
import numpy as np

from fastNLP.core.vocabulary import Vocabulary
from fastNLP.io.embed_loader import EmbedLoader


class TestEmbedLoader(unittest.TestCase):
    def test_case(self):
        vocab = Vocabulary()
        vocab.update(["the", "in", "I", "to", "of", "hahaha"])
        embedding = EmbedLoader().fast_load_embedding(50, "test/data_for_tests/glove.6B.50d_test.txt", vocab)
        self.assertEqual(tuple(embedding.shape), (len(vocab), 50))

    def test_load_with_vocab(self):
        vocab = Vocabulary()
        glove = "test/data_for_tests/glove.6B.50d_test.txt"
        word2vec = "test/data_for_tests/word2vec_test.txt"
        vocab.add_word('the')
        g_m = EmbedLoader.load_with_vocab(glove, vocab)
        self.assertEqual(g_m.shape, (3, 50))
        w_m = EmbedLoader.load_with_vocab(word2vec, vocab, normalize=True)
        self.assertEqual(w_m.shape, (3, 50))
        self.assertAlmostEqual(np.linalg.norm(w_m, axis=1).sum(), 3)

    def test_load_without_vocab(self):
        words = ['the', 'of', 'in', 'a', 'to', 'and']
        glove = "test/data_for_tests/glove.6B.50d_test.txt"
        word2vec = "test/data_for_tests/word2vec_test.txt"
        g_m, vocab = EmbedLoader.load_without_vocab(glove)
        self.assertEqual(g_m.shape, (8, 50))
        for word in words:
            self.assertIn(word, vocab)
        w_m, vocab = EmbedLoader.load_without_vocab(word2vec, normalize=True)
        self.assertEqual(w_m.shape, (8, 50))
        self.assertAlmostEqual(np.linalg.norm(w_m, axis=1).sum(), 8)
        for word in words:
            self.assertIn(word, vocab)
        # no unk
        w_m, vocab = EmbedLoader.load_without_vocab(word2vec, normalize=True, unknown=None)
        self.assertEqual(w_m.shape, (7, 50))
        self.assertAlmostEqual(np.linalg.norm(w_m, axis=1).sum(), 7)
        for word in words:
            self.assertIn(word, vocab)
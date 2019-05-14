import unittest
import numpy as np

from fastNLP import Vocabulary
from fastNLP.io import EmbedLoader


class TestEmbedLoader(unittest.TestCase):
    def test_load_with_vocab(self):
        vocab = Vocabulary()
        glove = "test/data_for_tests/glove.6B.50d_test.txt"
        word2vec = "test/data_for_tests/word2vec_test.txt"
        vocab.add_word('the')
        vocab.add_word('none')
        g_m = EmbedLoader.load_with_vocab(glove, vocab)
        self.assertEqual(g_m.shape, (4, 50))
        w_m = EmbedLoader.load_with_vocab(word2vec, vocab, normalize=True)
        self.assertEqual(w_m.shape, (4, 50))
        self.assertAlmostEqual(np.linalg.norm(w_m, axis=1).sum(), 4)
    
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
    
    def test_read_all_glove(self):
        pass
        # TODO
        # 这是可以运行的，但是总数少于行数，应该是由于glove有重复的word
        # path = '/where/to/read/full/glove'
        # init_embed, vocab = EmbedLoader.load_without_vocab(path, error='strict')
        # print(init_embed.shape)
        # print(init_embed.mean())
        # print(np.isnan(init_embed).sum())
        # print(len(vocab))

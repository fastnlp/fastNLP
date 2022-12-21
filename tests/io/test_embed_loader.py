import numpy as np

from fastNLP import Vocabulary
from fastNLP.io import EmbedLoader


class TestEmbedLoader:
    def test_load_with_vocab(self):
        vocab = Vocabulary()
        glove = "data_for_tests/embedding/small_static_embedding/glove.6B.50d_test.txt"
        word2vec = "data_for_tests/embedding/small_static_embedding/word2vec_test.txt"
        vocab.add_word('the')
        vocab.add_word('none')
        g_m = EmbedLoader.load_with_vocab(glove, vocab)
        assert(g_m.shape == (4, 50))
        w_m = EmbedLoader.load_with_vocab(word2vec, vocab, normalize=True)
        assert(w_m.shape ==(4, 50))
        assert np.allclose(np.linalg.norm(w_m, axis=1).sum(), 4)
    
    def test_load_without_vocab(self):
        words = ['the', 'of', 'in', 'a', 'to', 'and']
        glove = "data_for_tests/embedding/small_static_embedding/glove.6B.50d_test.txt"
        word2vec = "data_for_tests/embedding/small_static_embedding/word2vec_test.txt"
        g_m, vocab = EmbedLoader.load_without_vocab(glove)
        assert(g_m.shape == (8, 50))
        for word in words:
            assert(word in vocab)
        w_m, vocab = EmbedLoader.load_without_vocab(word2vec, normalize=True)
        assert(w_m.shape== (8, 50))
        assert np.allclose(np.linalg.norm(w_m, axis=1).sum(), 8)
        for word in words:
            assert(word in vocab)
        # no unk
        w_m, vocab = EmbedLoader.load_without_vocab(word2vec, normalize=True, unknown=None)
        assert(w_m.shape == (7, 50))
        assert np.allclose(np.linalg.norm(w_m, axis=1).sum(), 7)
        for word in words:
            assert(word in vocab)

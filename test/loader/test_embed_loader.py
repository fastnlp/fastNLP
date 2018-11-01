import unittest
import os

import torch

from fastNLP.loader.embed_loader import EmbedLoader
from fastNLP.core.vocabulary import Vocabulary


class TestEmbedLoader(unittest.TestCase):
    glove_path = './test/data_for_tests/glove.6B.50d_test.txt'
    pkl_path = './save'
    raw_texts = ["i am a cat",
                "this is a test of new batch",
                "ha ha",
                "I am a good boy .",
                "This is the most beautiful girl ."
                ]
    texts = [text.strip().split() for text in raw_texts]
    vocab = Vocabulary()
    vocab.update(texts)
    def test1(self):
        emb, _ = EmbedLoader.load_embedding(50, self.glove_path, 'glove', self.vocab, self.pkl_path)
        self.assertTrue(emb.shape[0] == (len(self.vocab)))
        self.assertTrue(emb.shape[1] == 50)
        os.remove(self.pkl_path)
    
    def test2(self):
        try:
            _ = EmbedLoader.load_embedding(100, self.glove_path, 'glove', self.vocab, self.pkl_path)
            self.fail(msg="load dismatch embedding")
        except ValueError:
            pass

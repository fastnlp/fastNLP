import unittest
from fastNLP import Vocabulary
from fastNLP.embeddings import BertEmbedding
import torch
import os

@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
class TestDownload(unittest.TestCase):
    def test_download(self):
        # import os
        vocab = Vocabulary().add_word_lst("This is a test .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='en')
        words = torch.LongTensor([[2, 3, 4, 0]])
        print(embed(words).size())

    def test_word_drop(self):
        vocab = Vocabulary().add_word_lst("This is a test .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='en', dropout=0.1, word_dropout=0.2)
        for i in range(10):
            words = torch.LongTensor([[2, 3, 4, 0]])
            print(embed(words).size())
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
        words = torch.LongTensor([[0, 1, 2]])
        print(embed(words).size())

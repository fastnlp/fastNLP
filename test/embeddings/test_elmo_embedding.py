
import unittest
from fastNLP import Vocabulary
from fastNLP.embeddings import ElmoEmbedding
import torch
import os

@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
class TestDownload(unittest.TestCase):
    def test_download_small(self):
        # import os
        vocab = Vocabulary().add_word_lst("This is a test .".split())
        elmo_embed = ElmoEmbedding(vocab, model_dir_or_name='en-small')
        words = torch.LongTensor([[0, 1, 2]])
        print(elmo_embed(words).size())


# 首先保证所有权重可以加载；上传权重；验证可以下载




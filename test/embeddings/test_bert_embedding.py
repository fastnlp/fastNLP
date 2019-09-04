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

        for pool_method in ['first', 'last', 'max', 'avg']:
            for include_cls_sep in [True, False]:
                embed = BertEmbedding(vocab, model_dir_or_name='en', pool_method=pool_method,
                                      include_cls_sep=include_cls_sep)
                print(embed(words).size())

    def test_word_drop(self):
        vocab = Vocabulary().add_word_lst("This is a test .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='en', dropout=0.1, word_dropout=0.2)
        for i in range(10):
            words = torch.LongTensor([[2, 3, 4, 0]])
            print(embed(words).size())


class TestBertEmbedding(unittest.TestCase):
    def test_bert_embedding_1(self):
        vocab = Vocabulary().add_word_lst("this is a test .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert')
        words = torch.LongTensor([[2, 3, 4, 0]])
        result = embed(words)
        self.assertEqual(result.size(), (1, 4, 16))

import unittest
from fastNLP import Vocabulary
from fastNLP.embeddings import BertEmbedding, BertWordPieceEncoder
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
        vocab = Vocabulary().add_word_lst("this is a test . [SEP] NotInBERT".split())
        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert', word_dropout=0.1)
        requires_grad = embed.requires_grad
        embed.requires_grad = not requires_grad
        embed.train()
        words = torch.LongTensor([[2, 3, 4, 0]])
        result = embed(words)
        self.assertEqual(result.size(), (1, 4, 16))

        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert', word_dropout=0.1,
                              only_use_pretrain_bpe=True)
        embed.eval()
        words = torch.LongTensor([[2, 3, 4, 0]])
        result = embed(words)
        self.assertEqual(result.size(), (1, 4, 16))


class TestBertWordPieceEncoder(unittest.TestCase):
    def test_bert_word_piece_encoder(self):
        embed = BertWordPieceEncoder(model_dir_or_name='test/data_for_tests/embedding/small_bert', word_dropout=0.1)
        from fastNLP import DataSet
        ds = DataSet({'words': ["this is a test . [SEP]".split()]})
        embed.index_datasets(ds, field_name='words')
        self.assertTrue(ds.has_field('word_pieces'))
        result = embed(torch.LongTensor([[1,2,3,4]]))

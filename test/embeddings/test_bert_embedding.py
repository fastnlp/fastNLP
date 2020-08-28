import unittest
from fastNLP import Vocabulary
from fastNLP.embeddings import BertEmbedding, BertWordPieceEncoder
import torch
import os
from fastNLP import DataSet


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

        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert', word_dropout=0.1)
        embed.eval()
        words = torch.LongTensor([[2, 3, 4, 0]])
        result = embed(words)
        self.assertEqual(result.size(), (1, 4, 16))

        # 自动截断而不报错
        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert', word_dropout=0.1,
                              auto_truncate=True)

        words = torch.LongTensor([[2, 3, 4, 1]*10,
                                  [2, 3]+[0]*38])
        result = embed(words)
        self.assertEqual(result.size(), (2, 40, 16))

    def test_save_load(self):
        bert_save_test = 'bert_save_test'
        try:
            os.makedirs(bert_save_test, exist_ok=True)
            vocab = Vocabulary().add_word_lst("this is a test . [SEP] NotInBERT".split())
            embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert', word_dropout=0.1,
                                  auto_truncate=True)

            embed.save(bert_save_test)
            load_embed = BertEmbedding.load(bert_save_test)
            words = torch.randint(len(vocab), size=(2, 20))
            embed.eval(), load_embed.eval()
            self.assertEqual((embed(words) - load_embed(words)).sum(), 0)

        finally:
            import shutil
            shutil.rmtree(bert_save_test)


class TestBertWordPieceEncoder(unittest.TestCase):
    def test_bert_word_piece_encoder(self):
        embed = BertWordPieceEncoder(model_dir_or_name='test/data_for_tests/embedding/small_bert', word_dropout=0.1)
        ds = DataSet({'words': ["this is a test . [SEP]".split()]})
        embed.index_datasets(ds, field_name='words')
        self.assertTrue(ds.has_field('word_pieces'))
        result = embed(torch.LongTensor([[1,2,3,4]]))

    def test_bert_embed_eq_bert_piece_encoder(self):
        ds = DataSet({'words': ["this is a texta model vocab".split(), 'this is'.split()]})
        encoder = BertWordPieceEncoder(model_dir_or_name='test/data_for_tests/embedding/small_bert')
        encoder.eval()
        encoder.index_datasets(ds, field_name='words')
        word_pieces = torch.LongTensor(ds['word_pieces'].get([0, 1]))
        word_pieces_res = encoder(word_pieces)

        vocab = Vocabulary()
        vocab.from_dataset(ds, field_name='words')
        vocab.index_dataset(ds, field_name='words', new_field_name='words')
        ds.set_input('words')
        words = torch.LongTensor(ds['words'].get([0, 1]))
        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert',
                              pool_method='first', include_cls_sep=True, pooled_cls=False, min_freq=1)
        embed.eval()
        words_res = embed(words)

        # 检查word piece什么的是正常work的
        self.assertEqual((word_pieces_res[0, :5]-words_res[0, :5]).sum(), 0)
        self.assertEqual((word_pieces_res[0, 6:]-words_res[0, 5:]).sum(), 0)
        self.assertEqual((word_pieces_res[1, :3]-words_res[1, :3]).sum(), 0)

    def test_save_load(self):
        bert_save_test = 'bert_save_test'
        try:
            os.makedirs(bert_save_test, exist_ok=True)
            embed = BertWordPieceEncoder(model_dir_or_name='test/data_for_tests/embedding/small_bert', word_dropout=0.0,
                                         layers='-2')
            ds = DataSet({'words': ["this is a test . [SEP]".split()]})
            embed.index_datasets(ds, field_name='words')
            self.assertTrue(ds.has_field('word_pieces'))
            words = torch.LongTensor([[1, 2, 3, 4]])
            embed.save(bert_save_test)
            load_embed = BertWordPieceEncoder.load(bert_save_test)
            embed.eval(), load_embed.eval()
            self.assertEqual((embed(words) - load_embed(words)).sum(), 0)
        finally:
            import shutil
            shutil.rmtree(bert_save_test)


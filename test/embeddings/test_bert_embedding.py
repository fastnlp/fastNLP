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

        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert', word_dropout=0.1,
                              only_use_pretrain_bpe=True)
        embed.eval()
        words = torch.LongTensor([[2, 3, 4, 0]])
        result = embed(words)
        self.assertEqual(result.size(), (1, 4, 16))

        # 自动截断而不报错
        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert', word_dropout=0.1,
                              only_use_pretrain_bpe=True, auto_truncate=True)

        words = torch.LongTensor([[2, 3, 4, 1]*10,
                                  [2, 3]+[0]*38])
        result = embed(words)
        self.assertEqual(result.size(), (2, 40, 16))

    def test_bert_embedding_2(self):
        # 测试only_use_pretrain_vocab与truncate_embed是否正常工作
        with open('test/data_for_tests/embedding/small_bert/vocab.txt', 'r', encoding='utf-8') as f:
            num_word = len(f.readlines())
        Embedding = BertEmbedding
        vocab = Vocabulary().add_word_lst("this is a texta and [SEP] NotInBERT".split())
        embed1 = Embedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert',
                              only_use_pretrain_bpe=True, truncate_embed=True, min_freq=1)
        embed_bpe_vocab_size = len(vocab)-1 + 2  # 排除NotInBERT, 额外加##a, [CLS]
        self.assertEqual(embed_bpe_vocab_size, len(embed1.model.tokenzier.vocab))

        embed2 = Embedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert',
                              only_use_pretrain_bpe=True, truncate_embed=False, min_freq=1)
        embed_bpe_vocab_size = num_word  # 排除NotInBERT
        self.assertEqual(embed_bpe_vocab_size, len(embed2.model.tokenzier.vocab))

        embed3 = Embedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert',
                              only_use_pretrain_bpe=False, truncate_embed=True, min_freq=1)
        embed_bpe_vocab_size = len(vocab)+2  # 新增##a, [CLS]
        self.assertEqual(embed_bpe_vocab_size, len(embed3.model.tokenzier.vocab))

        embed4 = Embedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert',
                              only_use_pretrain_bpe=False, truncate_embed=False, min_freq=1)
        embed_bpe_vocab_size = num_word+1  # 新增##a
        self.assertEqual(embed_bpe_vocab_size, len(embed4.model.tokenzier.vocab))

        # 测试各种情况下以下tensor的值是相等的
        embed1.eval()
        embed2.eval()
        embed3.eval()
        embed4.eval()
        tensor = torch.LongTensor([[vocab.to_index(w) for w in 'this is a texta and'.split()]])
        t1 = embed1(tensor)
        t2 = embed2(tensor)
        t3 = embed3(tensor)
        t4 = embed4(tensor)

        self.assertEqual((t1-t2).sum(), 0)
        self.assertEqual((t1-t3).sum(), 0)
        self.assertEqual((t1-t4).sum(), 0)


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
                              pool_method='first', include_cls_sep=True, pooled_cls=False)
        embed.eval()
        words_res = embed(words)

        # 检查word piece什么的是正常work的
        self.assertEqual((word_pieces_res[0, :5]-words_res[0, :5]).sum(), 0)
        self.assertEqual((word_pieces_res[0, 6:]-words_res[0, 5:]).sum(), 0)
        self.assertEqual((word_pieces_res[1, :3]-words_res[1, :3]).sum(), 0)
import unittest

import torch

from fastNLP.core import Vocabulary, Const
from fastNLP.models.bert import BertForSequenceClassification, BertForQuestionAnswering, \
    BertForTokenClassification, BertForMultipleChoice, BertForSentenceMatching
from fastNLP.embeddings.bert_embedding import BertEmbedding


class TestBert(unittest.TestCase):
    def test_bert_1(self):
        vocab = Vocabulary().add_word_lst("this is a test .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert',
                              include_cls_sep=True)

        model = BertForSequenceClassification(embed, 2)

        input_ids = torch.LongTensor([[1, 2, 3], [5, 6, 0]])

        pred = model(input_ids)
        self.assertTrue(isinstance(pred, dict))
        self.assertTrue(Const.OUTPUT in pred)
        self.assertEqual(tuple(pred[Const.OUTPUT].shape), (2, 2))

        pred = model(input_ids)
        self.assertTrue(isinstance(pred, dict))
        self.assertTrue(Const.OUTPUT in pred)
        self.assertEqual(tuple(pred[Const.OUTPUT].shape), (2, 2))

    def test_bert_1_w(self):
        vocab = Vocabulary().add_word_lst("this is a test .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert',
                              include_cls_sep=False)

        with self.assertWarns(Warning):
            model = BertForSequenceClassification(embed, 2)

            input_ids = torch.LongTensor([[1, 2, 3], [5, 6, 0]])

            pred = model.predict(input_ids)
            self.assertTrue(isinstance(pred, dict))
            self.assertTrue(Const.OUTPUT in pred)
            self.assertEqual(tuple(pred[Const.OUTPUT].shape), (2,))

    def test_bert_2(self):

        vocab = Vocabulary().add_word_lst("this is a test [SEP] .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert',
                              include_cls_sep=True)

        model = BertForMultipleChoice(embed, 2)

        input_ids = torch.LongTensor([[[2, 6, 7], [1, 6, 5]]])
        print(input_ids.size())

        pred = model(input_ids)
        self.assertTrue(isinstance(pred, dict))
        self.assertTrue(Const.OUTPUT in pred)
        self.assertEqual(tuple(pred[Const.OUTPUT].shape), (1, 2))

    def test_bert_2_w(self):

        vocab = Vocabulary().add_word_lst("this is a test [SEP] .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert',
                              include_cls_sep=False)

        with self.assertWarns(Warning):
            model = BertForMultipleChoice(embed, 2)

            input_ids = torch.LongTensor([[[2, 6, 7], [1, 6, 5]]])
            print(input_ids.size())

            pred = model.predict(input_ids)
            self.assertTrue(isinstance(pred, dict))
            self.assertTrue(Const.OUTPUT in pred)
            self.assertEqual(tuple(pred[Const.OUTPUT].shape), (1,))

    def test_bert_3(self):

        vocab = Vocabulary().add_word_lst("this is a test [SEP] .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert',
                              include_cls_sep=False)
        model = BertForTokenClassification(embed, 7)

        input_ids = torch.LongTensor([[1, 2, 3], [6, 5, 0]])

        pred = model(input_ids)
        self.assertTrue(isinstance(pred, dict))
        self.assertTrue(Const.OUTPUT in pred)
        self.assertEqual(tuple(pred[Const.OUTPUT].shape), (2, 3, 7))

    def test_bert_3_w(self):

        vocab = Vocabulary().add_word_lst("this is a test [SEP] .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert',
                              include_cls_sep=True)

        with self.assertWarns(Warning):
            model = BertForTokenClassification(embed, 7)

            input_ids = torch.LongTensor([[1, 2, 3], [6, 5, 0]])

            pred = model.predict(input_ids)
            self.assertTrue(isinstance(pred, dict))
            self.assertTrue(Const.OUTPUT in pred)
            self.assertEqual(tuple(pred[Const.OUTPUT].shape), (2, 3))

    def test_bert_4(self):

        vocab = Vocabulary().add_word_lst("this is a test [SEP] .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert',
                              include_cls_sep=True)
        model = BertForQuestionAnswering(embed)

        input_ids = torch.LongTensor([[1, 2, 3], [6, 5, 0]])

        pred = model(input_ids)
        self.assertTrue(isinstance(pred, dict))
        self.assertTrue(Const.OUTPUTS(0) in pred)
        self.assertTrue(Const.OUTPUTS(1) in pred)
        self.assertEqual(tuple(pred[Const.OUTPUTS(0)].shape), (2, 5))
        self.assertEqual(tuple(pred[Const.OUTPUTS(1)].shape), (2, 5))

        model = BertForQuestionAnswering(embed, 7)
        pred = model(input_ids)
        self.assertTrue(isinstance(pred, dict))
        self.assertEqual(len(pred), 7)

    def test_bert_4_w(self):

        vocab = Vocabulary().add_word_lst("this is a test [SEP] .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert',
                              include_cls_sep=False)

        with self.assertWarns(Warning):
            model = BertForQuestionAnswering(embed)

            input_ids = torch.LongTensor([[1, 2, 3], [6, 5, 0]])

            pred = model.predict(input_ids)
            self.assertTrue(isinstance(pred, dict))
            self.assertTrue(Const.OUTPUTS(1) in pred)
            self.assertEqual(tuple(pred[Const.OUTPUTS(1)].shape), (2,))

    def test_bert_5(self):

        vocab = Vocabulary().add_word_lst("this is a test [SEP] .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert',
                              include_cls_sep=True)
        model = BertForSentenceMatching(embed)

        input_ids = torch.LongTensor([[1, 2, 3], [6, 5, 0]])

        pred = model(input_ids)
        self.assertTrue(isinstance(pred, dict))
        self.assertTrue(Const.OUTPUT in pred)
        self.assertEqual(tuple(pred[Const.OUTPUT].shape), (2, 2))

    def test_bert_5_w(self):

        vocab = Vocabulary().add_word_lst("this is a test [SEP] .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='test/data_for_tests/embedding/small_bert',
                              include_cls_sep=False)

        with self.assertWarns(Warning):
            model = BertForSentenceMatching(embed)

            input_ids = torch.LongTensor([[1, 2, 3], [6, 5, 0]])

            pred = model.predict(input_ids)
            self.assertTrue(isinstance(pred, dict))
            self.assertTrue(Const.OUTPUT in pred)
            self.assertEqual(tuple(pred[Const.OUTPUT].shape), (2,))


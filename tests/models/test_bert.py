import unittest

import torch

from fastNLP.core import Vocabulary, Const
from fastNLP.models.bert import BertForSequenceClassification, BertForQuestionAnswering, \
    BertForTokenClassification, BertForMultipleChoice, BertForSentenceMatching
from fastNLP.embeddings.bert_embedding import BertEmbedding


class TestBert(unittest.TestCase):
    def test_bert_1(self):
        vocab = Vocabulary().add_word_lst("this is a test .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='tests/data_for_tests/embedding/small_bert',
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
        embed = BertEmbedding(vocab, model_dir_or_name='tests/data_for_tests/embedding/small_bert',
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
        embed = BertEmbedding(vocab, model_dir_or_name='tests/data_for_tests/embedding/small_bert',
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
        embed = BertEmbedding(vocab, model_dir_or_name='tests/data_for_tests/embedding/small_bert',
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
        embed = BertEmbedding(vocab, model_dir_or_name='tests/data_for_tests/embedding/small_bert',
                              include_cls_sep=False)
        model = BertForTokenClassification(embed, 7)

        input_ids = torch.LongTensor([[1, 2, 3], [6, 5, 0]])

        pred = model(input_ids)
        self.assertTrue(isinstance(pred, dict))
        self.assertTrue(Const.OUTPUT in pred)
        self.assertEqual(tuple(pred[Const.OUTPUT].shape), (2, 3, 7))

    def test_bert_3_w(self):

        vocab = Vocabulary().add_word_lst("this is a test [SEP] .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='tests/data_for_tests/embedding/small_bert',
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
        embed = BertEmbedding(vocab, model_dir_or_name='tests/data_for_tests/embedding/small_bert',
                              include_cls_sep=False)
        model = BertForQuestionAnswering(embed)

        input_ids = torch.LongTensor([[1, 2, 3], [6, 5, 0]])

        pred = model(input_ids)
        self.assertTrue(isinstance(pred, dict))
        self.assertTrue('pred_start' in pred)
        self.assertTrue('pred_end' in pred)
        self.assertEqual(tuple(pred['pred_start'].shape), (2, 3))
        self.assertEqual(tuple(pred['pred_end'].shape), (2, 3))

    def test_bert_for_question_answering_train(self):
        from fastNLP import CMRC2018Loss
        from fastNLP.io import CMRC2018BertPipe
        from fastNLP import Trainer

        data_bundle = CMRC2018BertPipe().process_from_file('tests/data_for_tests/io/cmrc')
        data_bundle.rename_field('chars', 'words')
        train_data = data_bundle.get_dataset('train')
        vocab = data_bundle.get_vocab('words')

        embed = BertEmbedding(vocab, model_dir_or_name='tests/data_for_tests/embedding/small_bert',
                              include_cls_sep=False, auto_truncate=True)
        model = BertForQuestionAnswering(embed)
        loss = CMRC2018Loss()

        trainer = Trainer(train_data, model, loss=loss, use_tqdm=False)
        trainer.train(load_best_model=False)

    def test_bert_5(self):

        vocab = Vocabulary().add_word_lst("this is a test [SEP] .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='tests/data_for_tests/embedding/small_bert',
                              include_cls_sep=True)
        model = BertForSentenceMatching(embed)

        input_ids = torch.LongTensor([[1, 2, 3], [6, 5, 0]])

        pred = model(input_ids)
        self.assertTrue(isinstance(pred, dict))
        self.assertTrue(Const.OUTPUT in pred)
        self.assertEqual(tuple(pred[Const.OUTPUT].shape), (2, 2))

    def test_bert_5_w(self):

        vocab = Vocabulary().add_word_lst("this is a test [SEP] .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='tests/data_for_tests/embedding/small_bert',
                              include_cls_sep=False)

        with self.assertWarns(Warning):
            model = BertForSentenceMatching(embed)

            input_ids = torch.LongTensor([[1, 2, 3], [6, 5, 0]])

            pred = model.predict(input_ids)
            self.assertTrue(isinstance(pred, dict))
            self.assertTrue(Const.OUTPUT in pred)
            self.assertEqual(tuple(pred[Const.OUTPUT].shape), (2,))


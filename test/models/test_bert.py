import unittest

import torch

from fastNLP.models.bert import *


class TestBert(unittest.TestCase):
    def test_bert_1(self):
        from fastNLP.core.const import Const

        model = BertForSequenceClassification(2)

        input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        pred = model(input_ids, token_type_ids, input_mask)
        self.assertTrue(isinstance(pred, dict))
        self.assertTrue(Const.OUTPUT in pred)
        self.assertEqual(tuple(pred[Const.OUTPUT].shape), (2, 2))

    def test_bert_2(self):
        from fastNLP.core.const import Const

        model = BertForMultipleChoice(2)

        input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        pred = model(input_ids, token_type_ids, input_mask)
        self.assertTrue(isinstance(pred, dict))
        self.assertTrue(Const.OUTPUT in pred)
        self.assertEqual(tuple(pred[Const.OUTPUT].shape), (1, 2))

    def test_bert_3(self):
        from fastNLP.core.const import Const

        model = BertForTokenClassification(7)

        input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        pred = model(input_ids, token_type_ids, input_mask)
        self.assertTrue(isinstance(pred, dict))
        self.assertTrue(Const.OUTPUT in pred)
        self.assertEqual(tuple(pred[Const.OUTPUT].shape), (2, 3, 7))

    def test_bert_4(self):
        from fastNLP.core.const import Const

        model = BertForQuestionAnswering()

        input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        pred = model(input_ids, token_type_ids, input_mask)
        self.assertTrue(isinstance(pred, dict))
        self.assertTrue(Const.OUTPUTS(0) in pred)
        self.assertTrue(Const.OUTPUTS(1) in pred)
        self.assertEqual(tuple(pred[Const.OUTPUTS(0)].shape), (2, 3))
        self.assertEqual(tuple(pred[Const.OUTPUTS(1)].shape), (2, 3))

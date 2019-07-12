
import unittest

import torch

from fastNLP.models.bert import BertModel


class TestBert(unittest.TestCase):
    def test_bert_1(self):
        from fastNLP.modules.encoder.bert import BertConfig
        config = BertConfig(32000)
        model = BertModel(config)

        input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
        for layer in all_encoder_layers:
            self.assertEqual(tuple(layer.shape), (2, 3, 768))
        self.assertEqual(tuple(pooled_output.shape), (2, 768))

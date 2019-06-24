
import torch
import torch.nn as nn

from fastNLP.core.const import Const
from fastNLP.models import BaseModel
from fastNLP.modules.encoder.bert import BertModel


class BertForNLI(BaseModel):
    # TODO: still in progress

    def __init__(self, class_num=3, bert_dir=None):
        super(BertForNLI, self).__init__()
        if bert_dir is not None:
            self.bert = BertModel.from_pretrained(bert_dir)
        else:
            self.bert = BertModel()
        hidden_size = self.bert.pooler.dense._parameters['bias'].size(-1)
        self.classifier = nn.Linear(hidden_size, class_num)

    def forward(self, words, seq_len1, seq_len2, target=None):
        """
        :param torch.Tensor words: [batch_size, seq_len] input_ids
        :param torch.Tensor seq_len1: [batch_size, seq_len] token_type_ids
        :param torch.Tensor seq_len2: [batch_size, seq_len] attention_mask
        :param torch.Tensor target: [batch]
        :return:
        """
        _, pooled_output = self.bert(words, seq_len1, seq_len2)
        logits = self.classifier(pooled_output)

        if target is not None:
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(logits, target)
            return {Const.OUTPUT: logits, Const.LOSS: loss}
        return {Const.OUTPUT: logits}

    def predict(self, words, seq_len1, seq_len2, target=None):
        return self.forward(words, seq_len1, seq_len2)


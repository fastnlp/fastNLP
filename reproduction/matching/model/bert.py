
import torch
import torch.nn as nn

from fastNLP.core.const import Const
from fastNLP.models.base_model import BaseModel
from fastNLP.embeddings import BertEmbedding


class BertForNLI(BaseModel):

    def __init__(self, bert_embed: BertEmbedding, class_num=3):
        super(BertForNLI, self).__init__()
        self.embed = bert_embed
        self.classifier = nn.Linear(self.embed.embedding_dim, class_num)

    def forward(self, words):
        """
        :param torch.Tensor words: [batch_size, seq_len] input_ids
        :return:
        """
        hidden = self.embed(words)
        logits = self.classifier(hidden)

        return {Const.OUTPUT: logits}

    def predict(self, words):
        logits = self.forward(words)[Const.OUTPUT]
        return {Const.OUTPUT: logits.argmax(dim=-1)}


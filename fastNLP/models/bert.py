"""
fastNLP提供了BERT应用到五个下游任务的模型代码，可以直接调用。这五个任务分别为

    - 文本分类任务： :class:`~fastNLP.models.BertForSequenceClassification`
    - Matching任务： :class:`~fastNLP.models.BertForSentenceMatching`
    - 多选任务： :class:`~fastNLP.models.BertForMultipleChoice`
    - 序列标注任务： :class:`~fastNLP.models.BertForTokenClassification`
    - 抽取式QA任务： :class:`~fastNLP.models.BertForQuestionAnswering`

每一个模型必须要传入一个名字为 `embed` 的 :class:`fastNLP.embeddings.BertEmbedding` ，这个参数包含了
:class:`fastNLP.modules.encoder.BertModel` ，是下游模型的编码器(encoder)。

除此以外，还需要传入一个数字，这个数字在不同下游任务模型上的意义如下::

    下游任务模型                     参数名称      含义
    BertForSequenceClassification  num_labels  文本分类类别数目，默认值为2
    BertForSentenceMatching        num_labels  Matching任务类别数目，默认值为2
    BertForMultipleChoice          num_choices 多选任务选项数目，默认值为2
    BertForTokenClassification     num_labels  序列标注标签数目，无默认值
    BertForQuestionAnswering       num_labels  抽取式QA列数，默认值为2(即第一列为start_span, 第二列为end_span)

最后还可以传入dropout的大小，默认值为0.1。

"""

__all__ = [
    "BertForSequenceClassification",
    "BertForSentenceMatching",
    "BertForMultipleChoice",
    "BertForTokenClassification",
    "BertForQuestionAnswering"
]

import warnings

import torch
from torch import nn

from .base_model import BaseModel
from ..core.const import Const
from ..core._logger import logger
from ..embeddings import BertEmbedding


class BertForSequenceClassification(BaseModel):
    """
    BERT model for classification.

    :param fastNLP.embeddings.BertEmbedding embed: 下游模型的编码器(encoder).
    :param int num_labels: 文本分类类别数目，默认值为2.
    :param float dropout: dropout的大小，默认值为0.1.
    """
    def __init__(self, embed: BertEmbedding, num_labels: int=2, dropout=0.1):
        super(BertForSequenceClassification, self).__init__()

        self.num_labels = num_labels
        self.bert = embed
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.embedding_dim, num_labels)

        if not self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = True
            warn_msg = "Bert for sequence classification excepts BertEmbedding `include_cls_sep` True, " \
                       "but got False. FastNLP has changed it to True."
            logger.warn(warn_msg)
            warnings.warn(warn_msg)

    def forward(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.Tensor [batch_size, num_labels]
        """
        hidden = self.dropout(self.bert(words))
        cls_hidden = hidden[:, 0]
        logits = self.classifier(cls_hidden)

        return {Const.OUTPUT: logits}

    def predict(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size]
        """
        logits = self.forward(words)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForSentenceMatching(BaseModel):
    """
    BERT model for sentence matching.

    :param fastNLP.embeddings.BertEmbedding embed: 下游模型的编码器(encoder).
    :param int num_labels: Matching任务类别数目，默认值为2.
    :param float dropout: dropout的大小，默认值为0.1.
    """
    def __init__(self, embed: BertEmbedding, num_labels: int=2, dropout=0.1):
        super(BertForSentenceMatching, self).__init__()
        self.num_labels = num_labels
        self.bert = embed
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.embedding_dim, num_labels)

        if not self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = True
            warn_msg = "Bert for sentence matching excepts BertEmbedding `include_cls_sep` True, " \
                       "but got False. FastNLP has changed it to True."
            logger.warn(warn_msg)
            warnings.warn(warn_msg)

    def forward(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.Tensor [batch_size, num_labels]
        """
        hidden = self.bert(words)
        cls_hidden = self.dropout(hidden[:, 0])
        logits = self.classifier(cls_hidden)

        return {Const.OUTPUT: logits}

    def predict(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size]
        """
        logits = self.forward(words)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForMultipleChoice(BaseModel):
    """
    BERT model for multiple choice.

    :param fastNLP.embeddings.BertEmbedding embed: 下游模型的编码器(encoder).
    :param int num_choices: 多选任务选项数目，默认值为2.
    :param float dropout: dropout的大小，默认值为0.1.
    """
    def __init__(self, embed: BertEmbedding, num_choices=2, dropout=0.1):
        super(BertForMultipleChoice, self).__init__()

        self.num_choices = num_choices
        self.bert = embed
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.embedding_dim, 1)

        if not self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = True
            warn_msg = "Bert for multiple choice excepts BertEmbedding `include_cls_sep` True, " \
                       "but got False. FastNLP has changed it to True."
            logger.warn(warn_msg)
            warnings.warn(warn_msg)

    def forward(self, words):
        """
        :param torch.LongTensor words: [batch_size, num_choices, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size, num_choices]
        """
        batch_size, num_choices, seq_len = words.size()

        input_ids = words.view(batch_size * num_choices, seq_len)
        hidden = self.bert(input_ids)
        pooled_output = self.dropout(hidden[:, 0])
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        return {Const.OUTPUT: reshaped_logits}

    def predict(self, words):
        """
        :param torch.LongTensor words: [batch_size, num_choices, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size]
        """
        logits = self.forward(words)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForTokenClassification(BaseModel):
    """
    BERT model for token classification.

    :param fastNLP.embeddings.BertEmbedding embed: 下游模型的编码器(encoder).
    :param int num_labels: 序列标注标签数目，无默认值.
    :param float dropout: dropout的大小，默认值为0.1.
    """
    def __init__(self, embed: BertEmbedding, num_labels, dropout=0.1):
        super(BertForTokenClassification, self).__init__()

        self.num_labels = num_labels
        self.bert = embed
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.embedding_dim, num_labels)

        if self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = False
            warn_msg = "Bert for token classification excepts BertEmbedding `include_cls_sep` False, " \
                       "but got True. FastNLP has changed it to False."
            logger.warn(warn_msg)
            warnings.warn(warn_msg)

    def forward(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.Tensor [batch_size, seq_len, num_labels]
        """
        sequence_output = self.bert(words)  # [batch_size, seq_len, embed_dim]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return {Const.OUTPUT: logits}

    def predict(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size, seq_len]
        """
        logits = self.forward(words)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForQuestionAnswering(BaseModel):
    """
    BERT model for classification.

    :param fastNLP.embeddings.BertEmbedding embed: 下游模型的编码器(encoder).
    :param int num_labels: 抽取式QA列数，默认值为2(即第一列为start_span, 第二列为end_span).
    """
    def __init__(self, embed: BertEmbedding, num_labels=2):
        super(BertForQuestionAnswering, self).__init__()

        self.bert = embed
        self.num_labels = num_labels
        self.qa_outputs = nn.Linear(self.bert.embedding_dim, self.num_labels)

        if not self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = True
            warn_msg = "Bert for question answering excepts BertEmbedding `include_cls_sep` True, " \
                       "but got False. FastNLP has changed it to True."
            logger.warn(warn_msg)
            warnings.warn(warn_msg)

    def forward(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: 一个包含num_labels个logit的dict，每一个logit的形状都是[batch_size, seq_len]
        """
        sequence_output = self.bert(words)
        logits = self.qa_outputs(sequence_output)  # [batch_size, seq_len, num_labels]

        return {Const.OUTPUTS(i): logits[:, :, i] for i in range(self.num_labels)}

    def predict(self, words):
        """
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: 一个包含num_labels个logit的dict，每一个logit的形状都是[batch_size]
        """
        logits = self.forward(words)
        return {Const.OUTPUTS(i): torch.argmax(logits[Const.OUTPUTS(i)], dim=-1) for i in range(self.num_labels)}

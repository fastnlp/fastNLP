r"""
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
from ..core._logger import logger
from ..core.const import Const
from ..embeddings.bert_embedding import BertEmbedding


class BertForSequenceClassification(BaseModel):
    r"""
    BERT model for classification.

    """
    def __init__(self, embed: BertEmbedding, num_labels: int=2, dropout=0.1):
        r"""
        
        :param fastNLP.embeddings.BertEmbedding embed: 下游模型的编码器(encoder).
        :param int num_labels: 文本分类类别数目，默认值为2.
        :param float dropout: dropout的大小，默认值为0.1.
        """
        super(BertForSequenceClassification, self).__init__()

        self.num_labels = num_labels
        self.bert = embed
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.embedding_dim, num_labels)

        if not self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = True
            warn_msg = "Bert for sequence classification excepts BertEmbedding `include_cls_sep` True, " \
                       "but got False. FastNLP has changed it to True."
            logger.warning(warn_msg)
            warnings.warn(warn_msg)

    def forward(self, words):
        r"""
        输入为 [[w1, w2, w3, ...], [...]], BERTEmbedding会在开头和结尾额外加入[CLS]与[SEP]
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.Tensor [batch_size, num_labels]
        """
        hidden = self.dropout(self.bert(words))
        cls_hidden = hidden[:, 0]
        logits = self.classifier(cls_hidden)
        if logits.size(-1) == 1:
            logits = logits.squeeze(-1)

        return {Const.OUTPUT: logits}

    def predict(self, words):
        r"""
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size]
        """
        logits = self.forward(words)[Const.OUTPUT]
        if self.num_labels > 1:
            return {Const.OUTPUT: torch.argmax(logits, dim=-1)}
        else:
            return {Const.OUTPUT: logits}


class BertForSentenceMatching(BaseModel):
    r"""
    BERT model for sentence matching.

    """
    def __init__(self, embed: BertEmbedding, num_labels: int=2, dropout=0.1):
        r"""
        
        :param fastNLP.embeddings.BertEmbedding embed: 下游模型的编码器(encoder).
        :param int num_labels: Matching任务类别数目，默认值为2.
        :param float dropout: dropout的大小，默认值为0.1.
        """
        super(BertForSentenceMatching, self).__init__()
        self.num_labels = num_labels
        self.bert = embed
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.embedding_dim, num_labels)

        if not self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = True
            warn_msg = "Bert for sentence matching excepts BertEmbedding `include_cls_sep` True, " \
                       "but got False. FastNLP has changed it to True."
            logger.warning(warn_msg)
            warnings.warn(warn_msg)

    def forward(self, words):
        r"""
        输入words的格式为 [sent1] + [SEP] + [sent2]（BertEmbedding会在开头加入[CLS]和在结尾加入[SEP]），输出为batch_size x num_labels

        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.Tensor [batch_size, num_labels]
        """
        hidden = self.bert(words)
        cls_hidden = self.dropout(hidden[:, 0])
        logits = self.classifier(cls_hidden)

        return {Const.OUTPUT: logits}

    def predict(self, words):
        r"""
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size]
        """
        logits = self.forward(words)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForMultipleChoice(BaseModel):
    r"""
    BERT model for multiple choice.

    """
    def __init__(self, embed: BertEmbedding, num_choices=2, dropout=0.1):
        r"""
        
        :param fastNLP.embeddings.BertEmbedding embed: 下游模型的编码器(encoder).
        :param int num_choices: 多选任务选项数目，默认值为2.
        :param float dropout: dropout的大小，默认值为0.1.
        """
        super(BertForMultipleChoice, self).__init__()

        self.num_choices = num_choices
        self.bert = embed
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.embedding_dim, 1)

        if not self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = True
            warn_msg = "Bert for multiple choice excepts BertEmbedding `include_cls_sep` True, " \
                       "but got False. FastNLP has changed it to True."
            logger.warning(warn_msg)
            warnings.warn(warn_msg)

    def forward(self, words):
        r"""
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
        r"""
        :param torch.LongTensor words: [batch_size, num_choices, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size]
        """
        logits = self.forward(words)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForTokenClassification(BaseModel):
    r"""
    BERT model for token classification.

    """
    def __init__(self, embed: BertEmbedding, num_labels, dropout=0.1):
        r"""
        
        :param fastNLP.embeddings.BertEmbedding embed: 下游模型的编码器(encoder).
        :param int num_labels: 序列标注标签数目，无默认值.
        :param float dropout: dropout的大小，默认值为0.1.
        """
        super(BertForTokenClassification, self).__init__()

        self.num_labels = num_labels
        self.bert = embed
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(self.bert.embedding_dim, num_labels)

        if self.bert.model.include_cls_sep:
            self.bert.model.include_cls_sep = False
            warn_msg = "Bert for token classification excepts BertEmbedding `include_cls_sep` False, " \
                       "but got True. FastNLP has changed it to False."
            logger.warning(warn_msg)
            warnings.warn(warn_msg)

    def forward(self, words):
        r"""
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.Tensor [batch_size, seq_len, num_labels]
        """
        sequence_output = self.bert(words)  # [batch_size, seq_len, embed_dim]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        return {Const.OUTPUT: logits}

    def predict(self, words):
        r"""
        :param torch.LongTensor words: [batch_size, seq_len]
        :return: { :attr:`fastNLP.Const.OUTPUT` : logits}: torch.LongTensor [batch_size, seq_len]
        """
        logits = self.forward(words)[Const.OUTPUT]
        return {Const.OUTPUT: torch.argmax(logits, dim=-1)}


class BertForQuestionAnswering(BaseModel):
    r"""
    用于做Q&A的Bert模型，如果是Squad2.0请将BertEmbedding的include_cls_sep设置为True，Squad1.0或CMRC则设置为False

    """
    def __init__(self, embed: BertEmbedding):
        r"""
        
        :param fastNLP.embeddings.BertEmbedding embed: 下游模型的编码器(encoder).
        :param int num_labels: 抽取式QA列数，默认值为2(即第一列为start_span, 第二列为end_span).
        """
        super(BertForQuestionAnswering, self).__init__()

        self.bert = embed
        self.qa_outputs = nn.Linear(self.bert.embedding_dim, 2)

    def forward(self, words):
        r"""
        输入words为question + [SEP] + [paragraph]，BERTEmbedding在之后会额外加入开头的[CLS]和结尾的[SEP]. note:
            如果BERTEmbedding中include_cls_sep=True，则输出的start和end index相对输入words会增加一位；如果为BERTEmbedding中
            include_cls_sep=False, 则输出start和end index的位置与输入words的顺序完全一致

        :param torch.LongTensor words: [batch_size, seq_len]
        :return: 一个包含num_labels个logit的dict，每一个logit的形状都是[batch_size, seq_len + 2]
        """
        sequence_output = self.bert(words)
        logits = self.qa_outputs(sequence_output)  # [batch_size, seq_len, num_labels]

        return {'pred_start': logits[:, :, 0], 'pred_end': logits[:, :, 1]}

    def predict(self, words):
        return self.forward(words)

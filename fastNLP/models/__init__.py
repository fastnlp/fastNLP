r"""
fastNLP 在 :mod:`~fastNLP.models` 模块中内置了如 :class:`~fastNLP.models.CNNText` 、
:class:`~fastNLP.models.SeqLabeling` 等完整的模型，以供用户直接使用。

.. todo::
    这些模型的介绍（与主页一致）


"""
__all__ = [
    "CNNText",

    "SeqLabeling",
    "AdvSeqLabel",
    "BiLSTMCRF",

    "ESIM",

    "StarTransEnc",
    "STSeqLabel",
    "STNLICls",
    "STSeqCls",

    "BiaffineParser",
    "GraphParser",

    "BertForSequenceClassification",
    "BertForSentenceMatching",
    "BertForMultipleChoice",
    "BertForTokenClassification",
    "BertForQuestionAnswering",

    "TransformerSeq2SeqModel",
    "LSTMSeq2SeqModel",
    "Seq2SeqModel",

    'SequenceGeneratorModel'
]

from .base_model import BaseModel
from .bert import BertForMultipleChoice, BertForQuestionAnswering, BertForSequenceClassification, \
    BertForTokenClassification, BertForSentenceMatching
from .biaffine_parser import BiaffineParser, GraphParser
from .cnn_text_classification import CNNText
from .sequence_labeling import SeqLabeling, AdvSeqLabel, BiLSTMCRF
from .snli import ESIM
from .star_transformer import StarTransEnc, STSeqCls, STNLICls, STSeqLabel
from .seq2seq_model import TransformerSeq2SeqModel, LSTMSeq2SeqModel, Seq2SeqModel
from .seq2seq_generator import SequenceGeneratorModel
import sys
from ..doc_utils import doc_process

doc_process(sys.modules[__name__])

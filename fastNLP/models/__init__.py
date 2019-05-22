"""
fastNLP 在 :mod:`~fastNLP.models` 模块中内置了如 :class:`~fastNLP.models.CNNText` 、
:class:`~fastNLP.models.SeqLabeling` 等完整的模型，以供用户直接使用。

.. todo::
    这些模型的介绍（与主页一致）


"""
__all__ = [
    "CNNText",
    
    "SeqLabeling",
    "AdvSeqLabel",
    
    "ESIM",
    
    "StarTransEnc",
    "STSeqLabel",
    "STNLICls",
    "STSeqCls",
    
    "BiaffineParser",
    "GraphParser"
]

from .base_model import BaseModel
from .bert import BertForMultipleChoice, BertForQuestionAnswering, BertForSequenceClassification, \
    BertForTokenClassification
from .biaffine_parser import BiaffineParser, GraphParser
from .cnn_text_classification import CNNText
from .sequence_labeling import SeqLabeling, AdvSeqLabel
from .snli import ESIM
from .star_transformer import StarTransEnc, STSeqCls, STNLICls, STSeqLabel

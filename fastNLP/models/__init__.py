"""
使用 fastNLP 实现的一系列常见模型，具体有：
TODO 详细介绍的表格，与主页相对应

"""
from .base_model import BaseModel
from .bert import BertForMultipleChoice, BertForQuestionAnswering, BertForSequenceClassification, \
    BertForTokenClassification
from .biaffine_parser import BiaffineParser, GraphParser
from .cnn_text_classification import CNNText
from .sequence_modeling import SeqLabeling, AdvSeqLabel
from .snli import ESIM
from .star_transformer import STSeqCls, STNLICls, STSeqLabel

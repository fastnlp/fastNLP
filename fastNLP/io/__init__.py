"""
用于IO的模块, 具体包括:

1. 用于读入 embedding 的 :doc:`EmbedLoader <fastNLP.io.embed_loader>` 类,

2. 用于读入不同格式数据的 :doc:`Loader <fastNLP.io.loader>` 类

3. 用于处理读入数据的 :doc:`Pipe <fastNLP.io.pipe>` 类

4. 用于保存和载入模型的类, 参考 :doc:`model_io文档</fastNLP.io.model_io>`

这些类的使用方法如下:
"""
__all__ = [
    'EmbedLoader',

    'DataBundle',
    'DataSetLoader',

    'YelpLoader',
    'YelpFullLoader',
    'YelpPolarityLoader',
    'IMDBLoader',
    'SSTLoader',
    'SST2Loader',

    'ConllLoader',
    'Conll2003Loader',
    'Conll2003NERLoader',
    'OntoNotesNERLoader',
    'CTBLoader',

    'Loader',
    'CSVLoader',
    'JsonLoader',

    'CWSLoader',

    'MNLILoader',
    "QuoraLoader",
    "SNLILoader",
    "QNLILoader",
    "RTELoader",

    "YelpFullPipe",
    "YelpPolarityPipe",
    "SSTPipe",
    "SST2Pipe",
    "IMDBPipe",

    "Conll2003NERPipe",
    "OntoNotesNERPipe",

    "MatchingBertPipe",
    "RTEBertPipe",
    "SNLIBertPipe",
    "QuoraBertPipe",
    "QNLIBertPipe",
    "MNLIBertPipe",
    "MatchingPipe",
    "RTEPipe",
    "SNLIPipe",
    "QuoraPipe",
    "QNLIPipe",
    "MNLIPipe",

    'ModelLoader',
    'ModelSaver',
]

from .embed_loader import EmbedLoader
from .base_loader import DataBundle, DataSetLoader
from .dataset_loader import CSVLoader, JsonLoader
from .model_io import ModelLoader, ModelSaver

from .loader import *
from .pipe import *

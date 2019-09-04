"""
用于IO的模块, 具体包括:

1. 用于读入 embedding 的 :doc:`EmbedLoader <fastNLP.io.embed_loader>` 类,

2. 用于读入不同格式数据的 :doc:`Loader <fastNLP.io.loader>` 类

3. 用于处理读入数据的 :doc:`Pipe <fastNLP.io.pipe>` 类

4. 用于保存和载入模型的类, 参考 :doc:`model_io文档</fastNLP.io.model_io>`

这些类的使用方法如下:
"""
__all__ = [
    'DataBundle',
    
    'EmbedLoader',
    
    'Loader',
    
    'YelpLoader',
    'YelpFullLoader',
    'YelpPolarityLoader',
    'IMDBLoader',
    'SSTLoader',
    'SST2Loader',
    "ChnSentiCorpLoader",

    'ConllLoader',
    'Conll2003Loader',
    'Conll2003NERLoader',
    'OntoNotesNERLoader',
    'CTBLoader',
    "MsraNERLoader",
    "WeiboNERLoader",
    "PeopleDailyNERLoader",

    'CSVLoader',
    'JsonLoader',

    'CWSLoader',

    'MNLILoader',
    "QuoraLoader",
    "SNLILoader",
    "QNLILoader",
    "RTELoader",

    "Pipe",

    "YelpFullPipe",
    "YelpPolarityPipe",
    "SSTPipe",
    "SST2Pipe",
    "IMDBPipe",
    "ChnSentiCorpPipe",

    "Conll2003Pipe",
    "Conll2003NERPipe",
    "OntoNotesNERPipe",
    "MsraNERPipe",
    "PeopleDailyPipe",
    "WeiboNERPipe",

    "CWSPipe",

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
from .data_bundle import DataBundle
from .model_io import ModelLoader, ModelSaver

from .loader import *
from .pipe import *

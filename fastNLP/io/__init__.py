r"""
用于IO的模块, 具体包括:

1. 用于读入 embedding 的 :mod:`EmbedLoader <fastNLP.io.embed_loader>` 类,

2. 用于读入不同格式数据的 :mod:`Loader <fastNLP.io.loader>` 类

3. 用于处理读入数据的 :mod:`Pipe <fastNLP.io.pipe>` 类

4. 用于保存和载入模型的类, 参考 :mod:`model_io模块 <fastNLP.io.model_io>`

这些类的使用方法如下:
"""
__all__ = [
    'DataBundle',
    
    'EmbedLoader',
    
    'Loader',
    
    'CLSBaseLoader',
    'AGsNewsLoader',
    'DBPediaLoader',
    'YelpFullLoader',
    'YelpPolarityLoader',
    'IMDBLoader',
    'SSTLoader',
    'SST2Loader',
    "ChnSentiCorpLoader",
    "THUCNewsLoader",
    "WeiboSenti100kLoader",

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
    "CNXNLILoader",
    "BQCorpusLoader",
    "LCQMCLoader",

    "CMRC2018Loader",

    "Pipe",

    "CLSBasePipe",
    "AGsNewsPipe",
    "DBPediaPipe",
    "YelpFullPipe",
    "YelpPolarityPipe",
    "SSTPipe",
    "SST2Pipe",
    "IMDBPipe",
    "ChnSentiCorpPipe",
    "THUCNewsPipe",
    "WeiboSenti100kPipe",

    "Conll2003Pipe",
    "Conll2003NERPipe",
    "OntoNotesNERPipe",
    "MsraNERPipe",
    "PeopleDailyPipe",
    "WeiboNERPipe",

    "CWSPipe",
    
    "Conll2003NERPipe",
    "OntoNotesNERPipe",
    "MsraNERPipe",
    "WeiboNERPipe",
    "PeopleDailyPipe",
    "Conll2003Pipe",
    
    "MatchingBertPipe",
    "RTEBertPipe",
    "SNLIBertPipe",
    "QuoraBertPipe",
    "QNLIBertPipe",
    "MNLIBertPipe",
    "CNXNLIBertPipe",
    "BQCorpusBertPipe",
    "LCQMCBertPipe",
    "MatchingPipe",
    "RTEPipe",
    "SNLIPipe",
    "QuoraPipe",
    "QNLIPipe",
    "MNLIPipe",
    "LCQMCPipe",
    "CNXNLIPipe",
    "BQCorpusPipe",
    "RenamePipe",
    "GranularizePipe",
    "MachingTruncatePipe",

    "CMRC2018BertPipe",

    'ModelLoader',
    'ModelSaver',

]

import sys

from .data_bundle import DataBundle
from .embed_loader import EmbedLoader
from .loader import *
from .model_io import ModelLoader, ModelSaver
from .pipe import *
from ..doc_utils import doc_process

doc_process(sys.modules[__name__])
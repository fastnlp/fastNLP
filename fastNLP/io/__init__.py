"""
用于IO的模块, 具体包括:

1. 用于读入 embedding 的 :doc:`EmbedLoader <fastNLP.io.embed_loader>` 类,

2. 用于读入不同格式数据的 :doc:`DataSetLoader <fastNLP.io.dataset_loader>` 类

3. 用于读入不同数据集并进行预处理的 :doc:`DataLoader <fastNLP.io.data_loader>` 类

4. 用于保存和载入模型的类, 参考 :doc:`model_io文档</fastNLP.io.model_io>`

这些类的使用方法如下:
"""
__all__ = [
    'EmbedLoader',

    'CSVLoader',
    'JsonLoader',

    'DataBundle',
    'DataSetLoader',

    'ConllLoader',
    'Conll2003Loader',
    'IMDBLoader',
    'MatchingLoader',
    'SNLILoader',
    'MNLILoader',
    'MTL16Loader',
    'PeopleDailyCorpusLoader',
    'QNLILoader',
    'QuoraLoader',
    'RTELoader',
    'SSTLoader',
    'SST2Loader',
    'YelpLoader',
    
    'ModelLoader',
    'ModelSaver',
]

from .embed_loader import EmbedLoader
from .base_loader import DataBundle, DataSetLoader
from .dataset_loader import CSVLoader, JsonLoader
from .model_io import ModelLoader, ModelSaver

from .data_loader import *

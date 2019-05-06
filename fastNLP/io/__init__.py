"""
用于IO的模块, 具体包括:

1. 用于读入 embedding 的 :doc:`EmbedLoader <fastNLP.io.embed_loader>` 类,

2. 用于读入数据的 :doc:`DataSetLoader <fastNLP.io.dataset_loader>` 类

3. 用于保存和载入模型的类, 参考 :doc:`Model-IO <fastNLP.io.model_io>`

这些类的使用方法可以在对应module的文档下查看.
"""
from .embed_loader import EmbedLoader
from .dataset_loader import DataSetLoader, CSVLoader, JsonLoader, ConllLoader, SNLILoader, SSTLoader, \
    PeopleDailyCorpusLoader, Conll2003Loader
from .model_io import ModelLoader as ModelLoader, ModelSaver as ModelSaver

__all__ = [
    'EmbedLoader',
    
    'DataSetLoader',
    'CSVLoader',
    'JsonLoader',
    'ConllLoader',
    'SNLILoader',
    'SSTLoader',
    'PeopleDailyCorpusLoader',
    'Conll2003Loader',
    
    'ModelLoader',
    'ModelSaver',
]

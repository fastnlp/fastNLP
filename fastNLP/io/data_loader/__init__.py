"""undocumented
.. warning::

    本模块在 `0.5.0版本` 中被废弃，由 :mod:`~fastNLP.io.loader`  和 :mod:`~fastNLP.io.pipe` 模块替代。

用于读数据集的模块, 可以读取文本分类、序列标注、Matching任务的数据集

这些模块的具体介绍如下，您可以通过阅读 :doc:`教程</tutorials/tutorial_2_load_dataset>` 来进行了解。
"""
__all__ = [
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
]


from .conll import ConllLoader, Conll2003Loader
from .imdb import IMDBLoader
from .matching import MatchingLoader
from .mnli import MNLILoader
from .mtl import MTL16Loader
from .people_daily import PeopleDailyCorpusLoader
from .qnli import QNLILoader
from .quora import QuoraLoader
from .rte import RTELoader
from .snli import SNLILoader
from .sst import SSTLoader, SST2Loader
from .yelp import YelpLoader

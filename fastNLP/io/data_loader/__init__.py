"""
用于读数据集的模块, 可以读取文本分类、序列标注、Matching任务的数据集

这些模块的使用方法如下:
"""
__all__ = [
    'ConllLoader',
    'Conll2003Loader',
    'IMDBLoader',
    'MatchingLoader',
    'MNLILoader',
    'MTL16Loader',
    'PeopleDailyCorpusLoader',
    'QNLILoader',
    'QuoraLoader',
    'RTELoader',
    'SSTLoader',
    'SST2Loader',
    'SNLILoader',
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

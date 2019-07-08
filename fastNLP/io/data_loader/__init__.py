"""
用于读数据集的模块, 具体包括:

这些模块的使用方法如下:
"""
__all__ = [
    'IMDBLoader',
    'MatchingLoader',
    'MNLILoader',
    'MTL16Loader',
    'QNLILoader',
    'QuoraLoader',
    'RTELoader',
    'SSTLoader',
    'SNLILoader',
    'YelpLoader',
]


from .imdb import IMDBLoader
from .matching import MatchingLoader
from .mnli import MNLILoader
from .mtl import MTL16Loader
from .qnli import QNLILoader
from .quora import QuoraLoader
from .rte import RTELoader
from .snli import SNLILoader
from .sst import SSTLoader
from .yelp import YelpLoader

"""
用于读数据集的模块, 具体包括:

这些模块的使用方法如下:
"""
__all__ = [
    'SSTLoader',

    'MatchingLoader',
    'SNLILoader',
    'MNLILoader',
    'QNLILoader',
    'QuoraLoader',
    'RTELoader',
]

from .sst import SSTLoader
from .matching import MatchingLoader, SNLILoader, \
    MNLILoader, QNLILoader, QuoraLoader, RTELoader

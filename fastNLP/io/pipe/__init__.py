"""
Pipe用于处理数据，所有的Pipe都包含一个 process(data_bundle) 方法，传入一个 :class:`~fastNLP.io.DataBundle` 类型的对象,
在传入 data_bundle 上进行原位修改，并将其返回； process_from_file(paths) 传入的文件路径，返回一个 :class:`~fastNLP.io.DataBundle` 。
process(data_bundle) 或者 process_from_file(paths)的返回 :class:`~fastNLP.io.DataBundle` 中的 :class:`~fastNLP.DataSet`
 一般都包含原文与转换为index的输入以及转换为index的target；除了 :class:`~fastNLP.DataSet` 之外，还会包含将field转为index时所建立的词表。

"""
__all__ = [
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
]

from .classification import YelpFullPipe, YelpPolarityPipe, SSTPipe, SST2Pipe, IMDBPipe
from .conll import Conll2003NERPipe, OntoNotesNERPipe
from .matching import MatchingBertPipe, RTEBertPipe, SNLIBertPipe, QuoraBertPipe, QNLIBertPipe, MNLIBertPipe, \
    MatchingPipe, RTEPipe, SNLIPipe, QuoraPipe, QNLIPipe, MNLIPipe

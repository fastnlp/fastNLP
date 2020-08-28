r"""
Pipe用于处理通过 Loader 读取的数据，所有的 Pipe 都包含 ``process`` 和 ``process_from_file`` 两种方法。
``process(data_bundle)`` 传入一个 :class:`~fastNLP.io.DataBundle` 类型的对象, 在传入的 `data_bundle` 上进行原位修改，并将其返回；
``process_from_file(paths)`` 传入的文件路径，返回一个 :class:`~fastNLP.io.DataBundle` 类型的对象。
``process(data_bundle)`` 或者 ``process_from_file(paths)`` 的返回 `data_bundle` 中的 :class:`~fastNLP.DataSet`
一般都包含原文与转换为index的输入以及转换为index的target；除了 :class:`~fastNLP.DataSet` 之外，
`data_bundle` 还会包含将field转为index时所建立的词表。

"""
__all__ = [
    "Pipe",
    
    "CWSPipe",
    
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
    
    "CoReferencePipe",

    "CMRC2018BertPipe"
]

from .classification import CLSBasePipe, YelpFullPipe, YelpPolarityPipe, SSTPipe, SST2Pipe, IMDBPipe, ChnSentiCorpPipe, THUCNewsPipe, \
    WeiboSenti100kPipe, AGsNewsPipe, DBPediaPipe
from .conll import Conll2003NERPipe, OntoNotesNERPipe, MsraNERPipe, WeiboNERPipe, PeopleDailyPipe
from .conll import Conll2003Pipe
from .coreference import CoReferencePipe
from .cws import CWSPipe
from .matching import MatchingBertPipe, RTEBertPipe, SNLIBertPipe, QuoraBertPipe, QNLIBertPipe, MNLIBertPipe, \
    MatchingPipe, RTEPipe, SNLIPipe, QuoraPipe, QNLIPipe, MNLIPipe, CNXNLIBertPipe, CNXNLIPipe, BQCorpusBertPipe, \
    LCQMCPipe, BQCorpusPipe, LCQMCBertPipe, RenamePipe, GranularizePipe, MachingTruncatePipe
from .pipe import Pipe
from .qa import CMRC2018BertPipe

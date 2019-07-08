
from ...core.const import Const

from .matching import MatchingLoader
from ..dataset_loader import CSVLoader


class QuoraLoader(MatchingLoader, CSVLoader):
    """
    别名：:class:`fastNLP.io.QuoraLoader` :class:`fastNLP.io.data_loader.QuoraLoader`

    读取MNLI数据集，读取的DataSet包含fields::

        words1: list(str)，第一句文本, premise
        words2: list(str), 第二句文本, hypothesis
        target: str, 真实标签

    数据来源:
    """

    def __init__(self, paths: dict=None):
        paths = paths if paths is not None else {
            'train': 'train.tsv',
            'dev': 'dev.tsv',
            'test': 'test.tsv',
        }
        MatchingLoader.__init__(self, paths=paths)
        CSVLoader.__init__(self, sep='\t', headers=(Const.TARGET, Const.INPUTS(0), Const.INPUTS(1), 'pairID'))

    def _load(self, path):
        ds = CSVLoader._load(self, path)
        return ds

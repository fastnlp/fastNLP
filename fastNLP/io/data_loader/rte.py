
from ...core.const import Const

from .matching import MatchingLoader
from ..dataset_loader import CSVLoader


class RTELoader(MatchingLoader, CSVLoader):
    """
    别名：:class:`fastNLP.io.RTELoader` :class:`fastNLP.io.data_loader.RTELoader`

    读取RTE数据集，读取的DataSet包含fields::

        words1: list(str)，第一句文本, premise
        words2: list(str), 第二句文本, hypothesis
        target: str, 真实标签

    数据来源:
    """

    def __init__(self, paths: dict=None):
        paths = paths if paths is not None else {
            'train': 'train.tsv',
            'dev': 'dev.tsv',
            'test': 'test.tsv'  # test set has not label
        }
        MatchingLoader.__init__(self, paths=paths)
        self.fields = {
            'sentence1': Const.INPUTS(0),
            'sentence2': Const.INPUTS(1),
            'label': Const.TARGET,
        }
        CSVLoader.__init__(self, sep='\t')

    def _load(self, path):
        ds = CSVLoader._load(self, path)

        for k, v in self.fields.items():
            if k in ds.get_field_names():
                ds.rename_field(k, v)
        for fields in ds.get_all_fields():
            if Const.INPUT in fields:
                ds.apply(lambda x: x[fields].strip().split(), new_field_name=fields)

        return ds

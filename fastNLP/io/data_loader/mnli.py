
from ...core.const import Const

from .matching import MatchingLoader
from ..dataset_loader import CSVLoader


class MNLILoader(MatchingLoader, CSVLoader):
    """
    别名：:class:`fastNLP.io.MNLILoader` :class:`fastNLP.io.data_loader.MNLILoader`

    读取MNLI数据集，读取的DataSet包含fields::

        words1: list(str)，第一句文本, premise

        words2: list(str), 第二句文本, hypothesis

        target: str, 真实标签

    数据来源:
    """

    def __init__(self, paths: dict=None):
        paths = paths if paths is not None else {
            'train': 'train.tsv',
            'dev_matched': 'dev_matched.tsv',
            'dev_mismatched': 'dev_mismatched.tsv',
            'test_matched': 'test_matched.tsv',
            'test_mismatched': 'test_mismatched.tsv',
            # 'test_0.9_matched': 'multinli_0.9_test_matched_unlabeled.txt',
            # 'test_0.9_mismatched': 'multinli_0.9_test_mismatched_unlabeled.txt',

            # test_0.9_mathed与mismatched是MNLI0.9版本的（数据来源：kaggle）
        }
        MatchingLoader.__init__(self, paths=paths)
        CSVLoader.__init__(self, sep='\t')
        self.fields = {
            'sentence1_binary_parse': Const.INPUTS(0),
            'sentence2_binary_parse': Const.INPUTS(1),
            'gold_label': Const.TARGET,
        }

    def _load(self, path):
        ds = CSVLoader._load(self, path)

        for k, v in self.fields.items():
            if k in ds.get_field_names():
                ds.rename_field(k, v)

        if Const.TARGET in ds.get_field_names():
            if ds[0][Const.TARGET] == 'hidden':
                ds.delete_field(Const.TARGET)

        parentheses_table = str.maketrans({'(': None, ')': None})

        ds.apply(lambda ins: ins[Const.INPUTS(0)].translate(parentheses_table).strip().split(),
                 new_field_name=Const.INPUTS(0))
        ds.apply(lambda ins: ins[Const.INPUTS(1)].translate(parentheses_table).strip().split(),
                 new_field_name=Const.INPUTS(1))
        if Const.TARGET in ds.get_field_names():
            ds.drop(lambda x: x[Const.TARGET] == '-')
        return ds


from ...core.const import Const

from .matching import MatchingLoader
from ..dataset_loader import JsonLoader


class SNLILoader(MatchingLoader, JsonLoader):
    """
    别名：:class:`fastNLP.io.SNLILoader` :class:`fastNLP.io.data_loader.SNLILoader`

    读取SNLI数据集，读取的DataSet包含fields::

        words1: list(str)，第一句文本, premise
        words2: list(str), 第二句文本, hypothesis
        target: str, 真实标签

    数据来源: https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    """

    def __init__(self, paths: dict=None):
        fields = {
            'sentence1_binary_parse': Const.INPUTS(0),
            'sentence2_binary_parse': Const.INPUTS(1),
            'gold_label': Const.TARGET,
        }
        paths = paths if paths is not None else {
            'train': 'snli_1.0_train.jsonl',
            'dev': 'snli_1.0_dev.jsonl',
            'test': 'snli_1.0_test.jsonl'}
        MatchingLoader.__init__(self, paths=paths)
        JsonLoader.__init__(self, fields=fields)

    def _load(self, path):
        ds = JsonLoader._load(self, path)

        parentheses_table = str.maketrans({'(': None, ')': None})

        ds.apply(lambda ins: ins[Const.INPUTS(0)].translate(parentheses_table).strip().split(),
                 new_field_name=Const.INPUTS(0))
        ds.apply(lambda ins: ins[Const.INPUTS(1)].translate(parentheses_table).strip().split(),
                 new_field_name=Const.INPUTS(1))
        ds.drop(lambda x: x[Const.TARGET] == '-')
        return ds

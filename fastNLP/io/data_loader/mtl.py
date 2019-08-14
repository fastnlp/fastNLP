
from typing import Union, Dict

from ..base_loader import DataBundle
from ..dataset_loader import CSVLoader
from ...core.vocabulary import Vocabulary, VocabularyOption
from ...core.const import Const
from ..utils import check_loader_paths


class MTL16Loader(CSVLoader):
    """
    别名：:class:`fastNLP.io.MTL16Loader` :class:`fastNLP.io.data_loader.MTL16Loader`

    读取MTL16数据集，DataSet包含以下fields:

        words: list(str), 需要分类的文本

        target: str, 文本的标签

    数据来源：https://pan.baidu.com/s/1c2L6vdA

    """

    def __init__(self):
        super(MTL16Loader, self).__init__(headers=(Const.TARGET, Const.INPUT), sep='\t')

    def _load(self, path):
        dataset = super(MTL16Loader, self)._load(path)
        dataset.apply(lambda x: x[Const.INPUT].lower().split(), new_field_name=Const.INPUT)
        if len(dataset) == 0:
            raise RuntimeError(f"{path} has no valid data.")

        return dataset

    def process(self,
                paths: Union[str, Dict[str, str]],
                src_vocab_opt: VocabularyOption = None,
                tgt_vocab_opt: VocabularyOption = None,):

        paths = check_loader_paths(paths)
        datasets = {}
        info = DataBundle()
        for name, path in paths.items():
            dataset = self.load(path)
            datasets[name] = dataset

        src_vocab = Vocabulary() if src_vocab_opt is None else Vocabulary(**src_vocab_opt)
        src_vocab.from_dataset(datasets['train'], field_name=Const.INPUT)
        src_vocab.index_dataset(*datasets.values(), field_name=Const.INPUT)

        tgt_vocab = Vocabulary(unknown=None, padding=None) \
            if tgt_vocab_opt is None else Vocabulary(**tgt_vocab_opt)
        tgt_vocab.from_dataset(datasets['train'], field_name=Const.TARGET)
        tgt_vocab.index_dataset(*datasets.values(), field_name=Const.TARGET)

        info.vocabs = {
            Const.INPUT: src_vocab,
            Const.TARGET: tgt_vocab
        }

        info.datasets = datasets

        for name, dataset in info.datasets.items():
            dataset.set_input(Const.INPUT)
            dataset.set_target(Const.TARGET)

        return info


from typing import Union, Dict

from ..embed_loader import EmbeddingOption, EmbedLoader
from ..base_loader import DataSetLoader, DataBundle
from ...core.vocabulary import VocabularyOption, Vocabulary
from ...core.dataset import DataSet
from ...core.instance import Instance
from ...core.const import Const

from ..utils import get_tokenizer


class IMDBLoader(DataSetLoader):
    """
    读取IMDB数据集，DataSet包含以下fields:

        words: list(str), 需要分类的文本
        target: str, 文本的标签

    """

    def __init__(self):
        super(IMDBLoader, self).__init__()
        self.tokenizer = get_tokenizer()

    def _load(self, path):
        dataset = DataSet()
        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split('\t')
                target = parts[0]
                words = self.tokenizer(parts[1].lower())
                dataset.append(Instance(words=words, target=target))

        if len(dataset) == 0:
            raise RuntimeError(f"{path} has no valid data.")

        return dataset
    
    def process(self,
                paths: Union[str, Dict[str, str]],
                src_vocab_opt: VocabularyOption = None,
                tgt_vocab_opt: VocabularyOption = None,
                char_level_op=False):

        datasets = {}
        info = DataBundle()
        for name, path in paths.items():
            dataset = self.load(path)
            datasets[name] = dataset

        def wordtochar(words):
            chars = []
            for word in words:
                word = word.lower()
                for char in word:
                    chars.append(char)
                chars.append('')
            chars.pop()
            return chars

        if char_level_op:
            for dataset in datasets.values():
                dataset.apply_field(wordtochar, field_name="words", new_field_name='chars')

        datasets["train"], datasets["dev"] = datasets["train"].split(0.1, shuffle=False)

        src_vocab = Vocabulary() if src_vocab_opt is None else Vocabulary(**src_vocab_opt)
        src_vocab.from_dataset(datasets['train'], field_name='words')

        src_vocab.index_dataset(*datasets.values(), field_name='words')

        tgt_vocab = Vocabulary(unknown=None, padding=None) \
            if tgt_vocab_opt is None else Vocabulary(**tgt_vocab_opt)
        tgt_vocab.from_dataset(datasets['train'], field_name='target')
        tgt_vocab.index_dataset(*datasets.values(), field_name='target')

        info.vocabs = {
            Const.INPUT: src_vocab,
            Const.TARGET: tgt_vocab
        }

        info.datasets = datasets

        for name, dataset in info.datasets.items():
            dataset.set_input(Const.INPUT)
            dataset.set_target(Const.TARGET)

        return info




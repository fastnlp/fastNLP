from fastNLP.io.embed_loader import EmbeddingOption, EmbedLoader
from fastNLP.core.vocabulary import VocabularyOption
from fastNLP.io.base_loader import DataSetLoader, DataInfo
from typing import Union, Dict, List, Iterator
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import Const
# from reproduction.utils import check_dataloader_paths
from functools import partial
import pandas as pd

class yelpLoader(DataSetLoader):
    """
    读取IMDB数据集，DataSet包含以下fields:

        words: list(str), 需要分类的文本
        target: str, 文本的标签


    """

    def __init__(self):
        super(yelpLoader, self).__init__()

    def _load(self, path):
        dataset = DataSet()
        data = pd.read_csv(path, header=None, sep=",").values
        for line in data:
            target = str(line[0])
            words = str(line[1]).lower().split()
            dataset.append(Instance(words=words, target=target))
        if len(dataset)==0:
            raise RuntimeError(f"{path} has no valid data.")

        return dataset
    
    def process(self,
                paths: Union[str, Dict[str, str]],
                src_vocab_opt: VocabularyOption = None,
                tgt_vocab_opt: VocabularyOption = None,
                src_embed_opt: EmbeddingOption = None):
        
        # paths = check_dataloader_paths(paths)
        datasets = {}
        info = DataInfo()
        for name, path in paths.items():
            dataset = self.load(path)
            datasets[name] = dataset

        datasets["train"], datasets["dev"] = datasets["train"].split(0.1, shuffle=False)

        src_vocab = Vocabulary() if src_vocab_opt is None else Vocabulary(**src_vocab_opt)
        src_vocab.from_dataset(datasets['train'], field_name='words')
        src_vocab.index_dataset(*datasets.values(), field_name='words')

        tgt_vocab = Vocabulary(unknown=None, padding=None) \
            if tgt_vocab_opt is None else Vocabulary(**tgt_vocab_opt)
        tgt_vocab.from_dataset(datasets['train'], field_name='target')
        tgt_vocab.index_dataset(*datasets.values(), field_name='target')

        info.vocabs = {
            "words": src_vocab,
            "target": tgt_vocab
        }

        info.datasets = datasets

        if src_embed_opt is not None:
            embed = EmbedLoader.load_with_vocab(**src_embed_opt, vocab=src_vocab)
            info.embeddings['words'] = embed

        for name, dataset in info.datasets.items():
            dataset.set_input("words")
            dataset.set_target("target")

        return info

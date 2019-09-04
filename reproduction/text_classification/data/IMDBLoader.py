from fastNLP.io.embed_loader import EmbeddingOption, EmbedLoader
from fastNLP.core.vocabulary import VocabularyOption
from fastNLP.io.data_bundle import DataSetLoader, DataBundle
from typing import Union, Dict, List, Iterator
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import Vocabulary
from fastNLP import Const
# from reproduction.utils import check_dataloader_paths
from functools import partial
from reproduction.utils import check_dataloader_paths, get_tokenizer

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

        if len(dataset)==0:
            raise RuntimeError(f"{path} has no valid data.")

        return dataset
    
    def process(self,
                paths: Union[str, Dict[str, str]],
                src_vocab_opt: VocabularyOption = None,
                tgt_vocab_opt: VocabularyOption = None,
                src_embed_opt: EmbeddingOption = None,
                char_level_op=False):
      
        datasets = {}
        info = DataBundle()
        paths = check_dataloader_paths(paths)
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



if __name__=="__main__":
    datapath = {"train": "/remote-home/ygwang/IMDB_data/train.csv",
                "test": "/remote-home/ygwang/IMDB_data/test.csv"}
    datainfo=IMDBLoader().process(datapath,char_level_op=True)
    #print(datainfo.datasets["train"])
    len_count = 0
    for instance in datainfo.datasets["train"]:
        len_count += len(instance["chars"])

    ave_len = len_count / len(datainfo.datasets["train"])
    print(ave_len)


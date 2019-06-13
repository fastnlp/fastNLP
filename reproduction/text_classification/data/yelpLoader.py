import ast
from fastNLP import DataSet, Instance, Vocabulary
from fastNLP.core.vocabulary import VocabularyOption
from fastNLP.io import JsonLoader
from fastNLP.io.base_loader import DataInfo
from fastNLP.io.embed_loader import EmbeddingOption
from fastNLP.io.file_reader import _read_json
from typing import Union, Dict
from reproduction.Star_transformer.datasets import EmbedLoader
from reproduction.utils import check_dataloader_paths


class yelpLoader(JsonLoader):
    
    """
    读取Yelp数据集, DataSet包含fields:
    
        review_id: str, 22 character unique review id
        user_id: str, 22 character unique user id
        business_id: str, 22 character business id
        useful: int, number of useful votes received
        funny: int, number of funny votes received
        cool: int, number of cool votes received
        date: str, date formatted YYYY-MM-DD
        words: list(str), 需要分类的文本
        target: str, 文本的标签
    
    数据来源: https://www.yelp.com/dataset/download
    
    :param fine_grained: 是否使用SST-5标准，若 ``False`` , 使用SST-2。Default: ``False``
    """
    
    def __init__(self, fine_grained=False):
        super(yelpLoader, self).__init__()
        tag_v = {'1.0': 'very negative', '2.0': 'negative', '3.0': 'neutral',
            '4.0': 'positive', '5.0': 'very positive'}
        if not fine_grained:
            tag_v['1.0'] = tag_v['2.0']
            tag_v['5.0'] = tag_v['4.0']
        self.fine_grained = fine_grained
        self.tag_v = tag_v
    
    def _load(self, path):
        ds = DataSet()
        for idx, d in _read_json(path, fields=self.fields_list, dropna=self.dropna):
            d = ast.literal_eval(d)
            d["words"] = d.pop("text").split()
            d["target"] = self.tag_v[str(d.pop("stars"))]
            ds.append(Instance(**d))
        return ds

    def process(self, paths: Union[str, Dict[str, str]], vocab_opt: VocabularyOption = None,
                embed_opt: EmbeddingOption = None):
        paths = check_dataloader_paths(paths)
        datasets = {}
        info = DataInfo()
        vocab = Vocabulary(min_freq=2) if vocab_opt is None else Vocabulary(**vocab_opt)
        for name, path in paths.items():
            dataset = self.load(path)
            datasets[name] = dataset
            vocab.from_dataset(dataset, field_name="words")
        info.vocabs = vocab
        info.datasets = datasets
        if embed_opt is not None:
            embed = EmbedLoader.load_with_vocab(**embed_opt, vocab=vocab)
            info.embeddings['words'] = embed
        return info


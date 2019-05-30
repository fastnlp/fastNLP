from typing import Iterable
from nltk import Tree
from ..base_loader import DataInfo, DataSetLoader
from ...core.vocabulary import VocabularyOption, Vocabulary
from ...core.dataset import DataSet
from ...core.instance import Instance
from ..embed_loader import EmbeddingOption, EmbedLoader


class SSTLoader(DataSetLoader):
    """
    别名：:class:`fastNLP.io.SSTLoader` :class:`fastNLP.io.dataset_loader.SSTLoader`

    读取SST数据集, DataSet包含fields::

        words: list(str) 需要分类的文本
        target: str 文本的标签

    数据来源: https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip

    :param subtree: 是否将数据展开为子树，扩充数据量. Default: ``False``
    :param fine_grained: 是否使用SST-5标准，若 ``False`` , 使用SST-2。Default: ``False``
    """

    def __init__(self, subtree=False, fine_grained=False):
        self.subtree = subtree

        tag_v = {'0': 'very negative', '1': 'negative', '2': 'neutral',
                 '3': 'positive', '4': 'very positive'}
        if not fine_grained:
            tag_v['0'] = tag_v['1']
            tag_v['4'] = tag_v['3']
        self.tag_v = tag_v

    def _load(self, path):
        """

        :param str path: 存储数据的路径
        :return: 一个 :class:`~fastNLP.DataSet` 类型的对象
        """
        datalist = []
        with open(path, 'r', encoding='utf-8') as f:
            datas = []
            for l in f:
                datas.extend([(s, self.tag_v[t])
                              for s, t in self._get_one(l, self.subtree)])
        ds = DataSet()
        for words, tag in datas:
            ds.append(Instance(words=words, target=tag))
        return ds

    @staticmethod
    def _get_one(data, subtree):
        tree = Tree.fromstring(data)
        if subtree:
            return [(t.leaves(), t.label()) for t in tree.subtrees()]
        return [(tree.leaves(), tree.label())]

    def process(self,
                paths,
                train_ds: Iterable[str] = None,
                src_vocab_op: VocabularyOption = None,
                tgt_vocab_op: VocabularyOption = None,
                src_embed_op: EmbeddingOption = None):
        input_name, target_name = 'words', 'target'
        src_vocab = Vocabulary() if src_vocab_op is None else Vocabulary(**src_vocab_op)
        tgt_vocab = Vocabulary() if tgt_vocab_op is None else Vocabulary(**tgt_vocab_op)

        info = DataInfo(datasets=self.load(paths))
        _train_ds = [info.datasets[name]
                     for name in train_ds] if train_ds else info.datasets.values()
        src_vocab.from_dataset(*_train_ds, field_name=input_name)
        tgt_vocab.from_dataset(*_train_ds, field_name=target_name)
        src_vocab.index_dataset(
            *info.datasets.values(),
            field_name=input_name, new_field_name=input_name)
        tgt_vocab.index_dataset(
            *info.datasets.values(),
            field_name=target_name, new_field_name=target_name)
        info.vocabs = {
            input_name: src_vocab,
            target_name: tgt_vocab
        }

        if src_embed_op is not None:
            src_embed_op.vocab = src_vocab
            init_emb = EmbedLoader.load_with_vocab(**src_embed_op)
            info.embeddings[input_name] = init_emb

        return info


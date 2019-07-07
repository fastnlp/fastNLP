from typing import Iterable
from nltk import Tree
import spacy
from ..base_loader import DataInfo, DataSetLoader
from ...core.vocabulary import VocabularyOption, Vocabulary
from ...core.dataset import DataSet
from ...core.instance import Instance
from ..utils import check_dataloader_paths, get_tokenizer


class SSTLoader(DataSetLoader):
    URL = 'https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip'
    DATA_DIR = 'sst/'

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
        self.tokenizer = get_tokenizer()

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

    def _get_one(self, data, subtree):
        tree = Tree.fromstring(data)
        if subtree:
            return [([x.text for x in self.tokenizer(' '.join(t.leaves()))], t.label()) for t in tree.subtrees() ]
        return [([x.text for x in self.tokenizer(' '.join(tree.leaves()))], tree.label())]

    def process(self,
                paths, train_subtree=True,
                src_vocab_op: VocabularyOption = None,
                tgt_vocab_op: VocabularyOption = None,):
        paths = check_dataloader_paths(paths)
        input_name, target_name = 'words', 'target'
        src_vocab = Vocabulary() if src_vocab_op is None else Vocabulary(**src_vocab_op)
        tgt_vocab = Vocabulary(unknown=None, padding=None) \
            if tgt_vocab_op is None else Vocabulary(**tgt_vocab_op)

        info = DataInfo()
        origin_subtree = self.subtree
        self.subtree = train_subtree
        info.datasets['train'] = self._load(paths['train'])
        self.subtree = origin_subtree
        for n, p in paths.items():
            if n != 'train':
                info.datasets[n] = self._load(p)

        src_vocab.from_dataset(
            info.datasets['train'],
            field_name=input_name,
            no_create_entry_dataset=[ds for n, ds in info.datasets.items() if n != 'train'])
        tgt_vocab.from_dataset(info.datasets['train'], field_name=target_name)

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

        return info


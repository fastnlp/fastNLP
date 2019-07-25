
from typing import Union, Dict
from nltk import Tree

from ..base_loader import DataBundle, DataSetLoader
from ..dataset_loader import CSVLoader
from ...core.vocabulary import VocabularyOption, Vocabulary
from ...core.dataset import DataSet
from ...core.const import Const
from ...core.instance import Instance
from ..utils import check_dataloader_paths, get_tokenizer


class SSTLoader(DataSetLoader):
    """
    别名：:class:`fastNLP.io.SSTLoader` :class:`fastNLP.io.data_loader.SSTLoader`

    读取SST数据集, DataSet包含fields::

        words: list(str) 需要分类的文本
        target: str 文本的标签

    数据来源: https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip

    :param subtree: 是否将数据展开为子树，扩充数据量. Default: ``False``
    :param fine_grained: 是否使用SST-5标准，若 ``False`` , 使用SST-2。Default: ``False``
    """

    URL = 'https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip'
    DATA_DIR = 'sst/'

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
            return [(self.tokenizer(' '.join(t.leaves())), t.label()) for t in tree.subtrees() ]
        return [(self.tokenizer(' '.join(tree.leaves())), tree.label())]

    def process(self,
                paths, train_subtree=True,
                src_vocab_op: VocabularyOption = None,
                tgt_vocab_op: VocabularyOption = None,):
        paths = check_dataloader_paths(paths)
        input_name, target_name = 'words', 'target'
        src_vocab = Vocabulary() if src_vocab_op is None else Vocabulary(**src_vocab_op)
        tgt_vocab = Vocabulary(unknown=None, padding=None) \
            if tgt_vocab_op is None else Vocabulary(**tgt_vocab_op)

        info = DataBundle()
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


class SST2Loader(CSVLoader):
    """
    别名：:class:`fastNLP.io.SST2Loader` :class:`fastNLP.io.data_loader.SST2Loader`

    数据来源 SST: https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8
    """

    def __init__(self):
        super(SST2Loader, self).__init__(sep='\t')
        self.tokenizer = get_tokenizer()
        self.field = {'sentence': Const.INPUT, 'label': Const.TARGET}

    def _load(self, path: str) -> DataSet:
        ds = super(SST2Loader, self)._load(path)
        for k, v in self.field.items():
            if k in ds.get_field_names():
                ds.rename_field(k, v)
        ds.apply(lambda x: self.tokenizer(x[Const.INPUT]), new_field_name=Const.INPUT)
        print("all count:", len(ds))
        return ds

    def process(self,
                paths: Union[str, Dict[str, str]],
                src_vocab_opt: VocabularyOption = None,
                tgt_vocab_opt: VocabularyOption = None,
                char_level_op=False):

        paths = check_dataloader_paths(paths)
        datasets = {}
        info = DataBundle()
        for name, path in paths.items():
            dataset = self.load(path)
            dataset.apply_field(lambda words:words.copy(), field_name='words', new_field_name='raw_words')
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

        input_name, target_name = Const.INPUT, Const.TARGET
        info.vocabs={}

        # 就分隔为char形式
        if char_level_op:
            for dataset in datasets.values():
                dataset.apply_field(wordtochar, field_name=Const.INPUT, new_field_name=Const.CHAR_INPUT)
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


"""
dataset_loader模块实现了许多 DataSetLoader, 用于读取不同格式的数据, 并返回 `DataSet` ,
得到的 :class:`~fastNLP.DataSet` 对象可以直接传入 :class:`~fastNLP.Trainer` 和 :class:`~fastNLP.Tester`, 用于模型的训练和测试。
以SNLI数据集为例::

    loader = SNLILoader()
    train_ds = loader.load('path/to/train')
    dev_ds = loader.load('path/to/dev')
    test_ds = loader.load('path/to/test')

    # ... do stuff
    
为 fastNLP 提供 DataSetLoader 的开发者请参考 :class:`~fastNLP.io.DataSetLoader` 的介绍。
"""
__all__ = [
    'CSVLoader',
    'JsonLoader',
    'ConllLoader',
    'MatchingLoader',
    'SNLILoader',
    'SSTLoader',
    'PeopleDailyCorpusLoader',
    'Conll2003Loader',
]

import os
from nltk import Tree
from typing import Union, Dict
from ..core.vocabulary import Vocabulary
from ..core.dataset import DataSet
from ..core.instance import Instance
from .file_reader import _read_csv, _read_json, _read_conll
from .base_loader import DataSetLoader, DataInfo
from .data_loader.sst import SSTLoader
from ..core.const import Const
from ..modules.encoder._bert import BertTokenizer


class PeopleDailyCorpusLoader(DataSetLoader):
    """
    别名：:class:`fastNLP.io.PeopleDailyCorpusLoader` :class:`fastNLP.io.dataset_loader.PeopleDailyCorpusLoader`

    读取人民日报数据集
    """

    def __init__(self, pos=True, ner=True):
        super(PeopleDailyCorpusLoader, self).__init__()
        self.pos = pos
        self.ner = ner

    def _load(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            sents = f.readlines()
        examples = []
        for sent in sents:
            if len(sent) <= 2:
                continue
            inside_ne = False
            sent_pos_tag = []
            sent_words = []
            sent_ner = []
            words = sent.strip().split()[1:]
            for word in words:
                if "[" in word and "]" in word:
                    ner_tag = "U"
                    print(word)
                elif "[" in word:
                    inside_ne = True
                    ner_tag = "B"
                    word = word[1:]
                elif "]" in word:
                    ner_tag = "L"
                    word = word[:word.index("]")]
                    if inside_ne is True:
                        inside_ne = False
                    else:
                        raise RuntimeError("only ] appears!")
                else:
                    if inside_ne is True:
                        ner_tag = "I"
                    else:
                        ner_tag = "O"
                tmp = word.split("/")
                token, pos = tmp[0], tmp[1]
                sent_ner.append(ner_tag)
                sent_pos_tag.append(pos)
                sent_words.append(token)
            example = [sent_words]
            if self.pos is True:
                example.append(sent_pos_tag)
            if self.ner is True:
                example.append(sent_ner)
            examples.append(example)
        return self.convert(examples)

    def convert(self, data):
        """

        :param data: python 内置对象
        :return: 一个 :class:`~fastNLP.DataSet` 类型的对象
        """
        data_set = DataSet()
        for item in data:
            sent_words = item[0]
            if self.pos is True and self.ner is True:
                instance = Instance(
                    words=sent_words, pos_tags=item[1], ner=item[2])
            elif self.pos is True:
                instance = Instance(words=sent_words, pos_tags=item[1])
            elif self.ner is True:
                instance = Instance(words=sent_words, ner=item[1])
            else:
                instance = Instance(words=sent_words)
            data_set.append(instance)
        data_set.apply(lambda ins: len(ins["words"]), new_field_name="seq_len")
        return data_set


class ConllLoader(DataSetLoader):
    """
    别名：:class:`fastNLP.io.ConllLoader` :class:`fastNLP.io.dataset_loader.ConllLoader`

    读取Conll格式的数据. 数据格式详见 http://conll.cemantix.org/2012/data.html

    列号从0开始, 每列对应内容为::

        Column  Type
        0       Document ID
        1       Part number
        2       Word number
        3       Word itself
        4       Part-of-Speech
        5       Parse bit
        6       Predicate lemma
        7       Predicate Frameset ID
        8       Word sense
        9       Speaker/Author
        10      Named Entities
        11:N    Predicate Arguments
        N       Coreference

    :param headers: 每一列数据的名称，需为List or Tuple  of str。``header`` 与 ``indexes`` 一一对应
    :param indexes: 需要保留的数据列下标，从0开始。若为 ``None`` ，则所有列都保留。Default: ``None``
    :param dropna: 是否忽略非法数据，若 ``False`` ，遇到非法数据时抛出 ``ValueError`` 。Default: ``False``
    """

    def __init__(self, headers, indexes=None, dropna=False):
        super(ConllLoader, self).__init__()
        if not isinstance(headers, (list, tuple)):
            raise TypeError(
                'invalid headers: {}, should be list of strings'.format(headers))
        self.headers = headers
        self.dropna = dropna
        if indexes is None:
            self.indexes = list(range(len(self.headers)))
        else:
            if len(indexes) != len(headers):
                raise ValueError
            self.indexes = indexes

    def _load(self, path):
        ds = DataSet()
        for idx, data in _read_conll(path, indexes=self.indexes, dropna=self.dropna):
            ins = {h: data[i] for i, h in enumerate(self.headers)}
            ds.append(Instance(**ins))
        return ds


class Conll2003Loader(ConllLoader):
    """
    别名：:class:`fastNLP.io.Conll2003Loader` :class:`fastNLP.io.dataset_loader.Conll2003Loader`

    读取Conll2003数据

    关于数据集的更多信息,参考:
    https://sites.google.com/site/ermasoftware/getting-started/ne-tagging-conll2003-data
    """

    def __init__(self):
        headers = [
            'tokens', 'pos', 'chunks', 'ner',
        ]
        super(Conll2003Loader, self).__init__(headers=headers)


def _cut_long_sentence(sent, max_sample_length=200):
    """
    将长于max_sample_length的sentence截成多段，只会在有空格的地方发生截断。
    所以截取的句子可能长于或者短于max_sample_length

    :param sent: str.
    :param max_sample_length: int.
    :return: list of str.
    """
    sent_no_space = sent.replace(' ', '')
    cutted_sentence = []
    if len(sent_no_space) > max_sample_length:
        parts = sent.strip().split()
        new_line = ''
        length = 0
        for part in parts:
            length += len(part)
            new_line += part + ' '
            if length > max_sample_length:
                new_line = new_line[:-1]
                cutted_sentence.append(new_line)
                length = 0
                new_line = ''
        if new_line != '':
            cutted_sentence.append(new_line[:-1])
    else:
        cutted_sentence.append(sent)
    return cutted_sentence


class JsonLoader(DataSetLoader):
    """
    别名：:class:`fastNLP.io.JsonLoader` :class:`fastNLP.io.dataset_loader.JsonLoader`

    读取json格式数据.数据必须按行存储,每行是一个包含各类属性的json对象

    :param dict fields: 需要读入的json属性名称, 和读入后在DataSet中存储的field_name
        ``fields`` 的 `key` 必须是json对象的属性名. ``fields`` 的 `value` 为读入后在DataSet存储的 `field_name` ,
        `value` 也可为 ``None`` , 这时读入后的 `field_name` 与json对象对应属性同名
        ``fields`` 可为 ``None`` , 这时,json对象所有属性都保存在DataSet中. Default: ``None``
    :param bool dropna: 是否忽略非法数据,若 ``True`` 则忽略,若 ``False`` ,在遇到非法数据时,抛出 ``ValueError`` .
        Default: ``False``
    """

    def __init__(self, fields=None, dropna=False):
        super(JsonLoader, self).__init__()
        self.dropna = dropna
        self.fields = None
        self.fields_list = None
        if fields:
            self.fields = {}
            for k, v in fields.items():
                self.fields[k] = k if v is None else v
            self.fields_list = list(self.fields.keys())

    def _load(self, path):
        ds = DataSet()
        for idx, d in _read_json(path, fields=self.fields_list, dropna=self.dropna):
            if self.fields:
                ins = {self.fields[k]: v for k, v in d.items()}
            else:
                ins = d
            ds.append(Instance(**ins))
        return ds


class MatchingLoader(DataSetLoader):
    """
        别名：:class:`fastNLP.io.MatchingLoader` :class:`fastNLP.io.dataset_loader.MatchingLoader`

        读取Matching数据集，根据数据集做预处理并返回DataInfo。

        数据来源:
        SNLI: https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    """

    def __init__(self, data_format: str='snli', for_model: str='esim', bert_dir=None):
        super(MatchingLoader, self).__init__()
        self.data_format = data_format.lower()
        self.for_model = for_model.lower()
        self.bert_dir = bert_dir

    def _load(self, path: str) -> DataSet:
        raise NotImplementedError

    def process(self, paths: Union[str, Dict[str, str]], **options) -> DataInfo:
        if isinstance(paths, str):
            paths = {'train': paths}

        data_set = {}
        for n, p in paths.items():
            if self.data_format == 'snli':
                data = self._load_snli(p)
            else:
                raise RuntimeError(f'Your data format is {self.data_format}, '
                                   f'Please choose data format from [snli]')

            if self.for_model == 'esim':
                data = self._for_esim(data)
            elif self.for_model == 'bert':
                data = self._for_bert(data, self.bert_dir)
            else:
                raise RuntimeError(f'Your model is {self.data_format}, '
                                   f'Please choose from [esim, bert]')

            data_set[n] = data
            print(f'successfully load {n} set!')

        if not hasattr(self, 'vocab'):
            raise RuntimeError(f'There is NOT vocab attribute built!')
        if not hasattr(self, 'label_vocab'):
            raise RuntimeError(f'There is NOT label vocab attribute built!')

        if self.for_model != 'bert':
            from fastNLP.modules.encoder.embedding import StaticEmbedding
            embedding = StaticEmbedding(self.vocab, model_dir_or_name='en')

        data_info = DataInfo(vocabs={'vocab': self.vocab, 'target_vocab': self.label_vocab},
                             embeddings={'glove': embedding} if self.for_model != 'bert' else None,
                             datasets=data_set)

        return data_info

    @staticmethod
    def _load_snli(path: str) -> DataSet:
        """
        读取SNLI数据集

        数据来源: https://nlp.stanford.edu/projects/snli/snli_1.0.zip
        :param str path: 数据集路径
        :return:
        """
        raw_ds = JsonLoader(
            fields={
                'sentence1_parse': Const.INPUTS(0),
                'sentence2_parse': Const.INPUTS(1),
                'gold_label': Const.TARGET,
            }
        )._load(path)
        return raw_ds

    def _for_esim(self, raw_ds: DataSet):
        if self.data_format == 'snli' or self.data_format == 'mnli':
            def parse_tree(x):
                t = Tree.fromstring(x)
                return t.leaves()

            raw_ds.apply(lambda ins: parse_tree(
                ins[Const.INPUTS(0)]), new_field_name=Const.INPUTS(0))
            raw_ds.apply(lambda ins: parse_tree(
                ins[Const.INPUTS(1)]), new_field_name=Const.INPUTS(1))
            raw_ds.drop(lambda x: x[Const.TARGET] == '-')

        if not hasattr(self, 'vocab'):
            self.vocab = Vocabulary().from_dataset(raw_ds, [Const.INPUTS(0), Const.INPUTS(1)])
        if not hasattr(self, 'label_vocab'):
            self.label_vocab = Vocabulary(padding=None, unknown=None).from_dataset(raw_ds, field_name=Const.TARGET)

        raw_ds.apply(lambda ins: [self.vocab.to_index(w) for w in ins[Const.INPUTS(0)]], new_field_name=Const.INPUTS(0))
        raw_ds.apply(lambda ins: [self.vocab.to_index(w) for w in ins[Const.INPUTS(1)]], new_field_name=Const.INPUTS(1))
        raw_ds.apply(lambda ins: self.label_vocab.to_index(Const.TARGET), new_field_name=Const.TARGET)

        raw_ds.set_input(Const.INPUTS(0), Const.INPUTS(1))
        raw_ds.set_target(Const.TARGET)

        return raw_ds

    def _for_bert(self, raw_ds: DataSet, bert_dir: str):
        if self.data_format == 'snli' or self.data_format == 'mnli':
            def parse_tree(x):
                t = Tree.fromstring(x)
                return t.leaves()

            raw_ds.apply(lambda ins: parse_tree(
                ins[Const.INPUTS(0)]), new_field_name=Const.INPUTS(0))
            raw_ds.apply(lambda ins: parse_tree(
                ins[Const.INPUTS(1)]), new_field_name=Const.INPUTS(1))
            raw_ds.drop(lambda x: x[Const.TARGET] == '-')

        tokenizer = BertTokenizer.from_pretrained(bert_dir)

        vocab = Vocabulary(padding=None, unknown=None)
        with open(os.path.join(bert_dir, 'vocab.txt')) as f:
            lines = f.readlines()
        vocab_list = []
        for line in lines:
            vocab_list.append(line.strip())
        vocab.add_word_lst(vocab_list)
        vocab.build_vocab()
        vocab.padding = '[PAD]'
        vocab.unknown = '[UNK]'

        if not hasattr(self, 'vocab'):
            self.vocab = vocab
        else:
            for w, idx in self.vocab:
                if vocab[w] != idx:
                    raise AttributeError(f"{self.__class__.__name__} has ")

        for i in range(2):
            raw_ds.apply(lambda x: tokenizer.tokenize(" ".join(x[Const.INPUTS(i)])), new_field_name=Const.INPUTS(i))
        raw_ds.apply(lambda x: ['[CLS]'] + x[Const.INPUTS(0)] + ['[SEP]'] + x[Const.INPUTS(1)] + ['[SEP]'],
                     new_field_name=Const.INPUT)
        raw_ds.apply(lambda x: [0] * (len(x[Const.INPUTS(0)]) + 2) + [1] * (len(x[Const.INPUTS(1)]) + 1),
                     new_field_name=Const.INPUT_LENS(0))
        raw_ds.apply(lambda x: [1] * len(x[Const.INPUT_LENS(0)]), new_field_name=Const.INPUT_LENS(1))

        max_len = 512
        raw_ds.apply(lambda x: x[Const.INPUT][: max_len], new_field_name=Const.INPUT)
        raw_ds.apply(lambda x: [self.vocab.to_index(w) for w in x[Const.INPUT]], new_field_name=Const.INPUT)
        raw_ds.apply(lambda x: x[Const.INPUT_LENS(0)][: max_len], new_field_name=Const.INPUT_LENS(0))
        raw_ds.apply(lambda x: x[Const.INPUT_LENS(1)][: max_len], new_field_name=Const.INPUT_LENS(1))

        if not hasattr(self, 'label_vocab'):
            self.label_vocab = Vocabulary(padding=None, unknown=None)
            self.label_vocab.from_dataset(raw_ds, field_name=Const.TARGET)
        raw_ds.apply(lambda x: self.label_vocab.to_index(x[Const.TARGET]), new_field_name=Const.TARGET)

        raw_ds.set_input(Const.INPUT, Const.INPUT_LENS(0), Const.INPUT_LENS(1))
        raw_ds.set_target(Const.TARGET)


class SNLILoader(JsonLoader):
    """
    别名：:class:`fastNLP.io.SNLILoader` :class:`fastNLP.io.dataset_loader.SNLILoader`

    读取SNLI数据集，读取的DataSet包含fields::

        words1: list(str)，第一句文本, premise
        words2: list(str), 第二句文本, hypothesis
        target: str, 真实标签

    数据来源: https://nlp.stanford.edu/projects/snli/snli_1.0.zip
    """

    def __init__(self):
        fields = {
            'sentence1_parse': Const.INPUTS(0),
            'sentence2_parse': Const.INPUTS(1),
            'gold_label': Const.TARGET,
        }
        super(SNLILoader, self).__init__(fields=fields)

    def _load(self, path):
        ds = super(SNLILoader, self)._load(path)

        def parse_tree(x):
            t = Tree.fromstring(x)
            return t.leaves()

        ds.apply(lambda ins: parse_tree(
            ins[Const.INPUTS(0)]), new_field_name=Const.INPUTS(0))
        ds.apply(lambda ins: parse_tree(
            ins[Const.INPUTS(1)]), new_field_name=Const.INPUTS(1))
        ds.drop(lambda x: x[Const.TARGET] == '-')
        return ds


class CSVLoader(DataSetLoader):
    """
    别名：:class:`fastNLP.io.CSVLoader` :class:`fastNLP.io.dataset_loader.CSVLoader`

    读取CSV格式的数据集。返回 ``DataSet``

    :param List[str] headers: CSV文件的文件头.定义每一列的属性名称,即返回的DataSet中`field`的名称
        若为 ``None`` ,则将读入文件的第一行视作 ``headers`` . Default: ``None``
    :param str sep: CSV文件中列与列之间的分隔符. Default: ","
    :param bool dropna: 是否忽略非法数据,若 ``True`` 则忽略,若 ``False`` ,在遇到非法数据时,抛出 ``ValueError`` .
        Default: ``False``
    """

    def __init__(self, headers=None, sep=",", dropna=False):
        self.headers = headers
        self.sep = sep
        self.dropna = dropna

    def _load(self, path):
        ds = DataSet()
        for idx, data in _read_csv(path, headers=self.headers,
                                   sep=self.sep, dropna=self.dropna):
            ds.append(Instance(**data))
        return ds


def _add_seg_tag(data):
    """

    :param data: list of ([word], [pos], [heads], [head_tags])
    :return: list of ([word], [pos])
    """

    _processed = []
    for word_list, pos_list, _, _ in data:
        new_sample = []
        for word, pos in zip(word_list, pos_list):
            if len(word) == 1:
                new_sample.append((word, 'S-' + pos))
            else:
                new_sample.append((word[0], 'B-' + pos))
                for c in word[1:-1]:
                    new_sample.append((c, 'M-' + pos))
                new_sample.append((word[-1], 'E-' + pos))
        _processed.append(list(map(list, zip(*new_sample))))
    return _processed

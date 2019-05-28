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
    'DataInfo',
    'DataSetLoader',
    'CSVLoader',
    'JsonLoader',
    'ConllLoader',
    'SNLILoader',
    'SSTLoader',
    'PeopleDailyCorpusLoader',
    'Conll2003Loader',
]

from nltk.tree import Tree

from ..core.dataset import DataSet
from ..core.instance import Instance
from .file_reader import _read_csv, _read_json, _read_conll
from typing import Union, Dict
import os


def _download_from_url(url, path):
    try:
        from tqdm.auto import tqdm
    except:
        from ..core.utils import _pseudo_tqdm as tqdm
    import requests
    
    """Download file"""
    r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
    chunk_size = 16 * 1024
    total_size = int(r.headers.get('Content-length', 0))
    with open(path, "wb") as file, \
            tqdm(total=total_size, unit='B', unit_scale=1, desc=path.split('/')[-1]) as t:
        for chunk in r.iter_content(chunk_size):
            if chunk:
                file.write(chunk)
                t.update(len(chunk))
    return


def _uncompress(src, dst):
    import zipfile
    import gzip
    import tarfile
    import os
    
    def unzip(src, dst):
        with zipfile.ZipFile(src, 'r') as f:
            f.extractall(dst)
    
    def ungz(src, dst):
        with gzip.open(src, 'rb') as f, open(dst, 'wb') as uf:
            length = 16 * 1024  # 16KB
            buf = f.read(length)
            while buf:
                uf.write(buf)
                buf = f.read(length)
    
    def untar(src, dst):
        with tarfile.open(src, 'r:gz') as f:
            f.extractall(dst)
    
    fn, ext = os.path.splitext(src)
    _, ext_2 = os.path.splitext(fn)
    if ext == '.zip':
        unzip(src, dst)
    elif ext == '.gz' and ext_2 != '.tar':
        ungz(src, dst)
    elif (ext == '.gz' and ext_2 == '.tar') or ext_2 == '.tgz':
        untar(src, dst)
    else:
        raise ValueError('unsupported file {}'.format(src))


class DataInfo:
    """
    经过处理的数据信息，包括一系列数据集（比如：分开的训练集、验证集和测试集）及它们所用的词表和词嵌入。

    :param vocabs: 从名称(字符串)到 :class:`~fastNLP.Vocabulary` 类型的dict
    :param embeddings: 从名称(字符串)到一系列 embedding 的dict，参考 :class:`~fastNLP.io.EmbedLoader`
    :param datasets: 从名称(字符串)到 :class:`~fastNLP.DataSet` 类型的dict
    """
    
    def __init__(self, vocabs: dict = None, embeddings: dict = None, datasets: dict = None):
        self.vocabs = vocabs or {}
        self.embeddings = embeddings or {}
        self.datasets = datasets or {}


class DataSetLoader:
    """
    别名：:class:`fastNLP.io.DataSetLoader` :class:`fastNLP.io.dataset_loader.DataSetLoader`

    定义了各种 DataSetLoader (针对特定数据上的特定任务) 所需的API 接口，开发者应该继承它实现各种的 DataSetLoader。
    
    开发者至少应该编写如下内容:
    
    - _load 函数：从一个数据文件中读取数据到一个 :class:`~fastNLP.DataSet`
    - load 函数（可以使用基类的方法）：从一个或多个数据文件中读取数据到一个或多个 :class:`~fastNLP.DataSet`
    - process 函数：一个或多个从数据文件中读取数据，并处理成可以训练的一个或多个 :class:`~fastNLP.DataSet`
    
    **process 函数中可以 调用load 函数或 _load 函数**
    
    """
    
    def _download(self, url: str, path: str, uncompress=True) -> str:
        """
        
        从 ``url`` 下载数据到 ``path``， 如果 ``uncompress`` 为 ``True`` ，自动解压。

        :param url: 下载的网站
        :param path: 下载到的目录
        :param uncompress:  是否自动解压缩
        :return: 数据的存放路径
        """
        pdir = os.path.dirname(path)
        os.makedirs(pdir, exist_ok=True)
        _download_from_url(url, path)
        if uncompress:
            dst = os.path.join(pdir, 'data')
            _uncompress(path, dst)
            return dst
        return path
    
    def load(self, paths: Union[str, Dict[str, str]]) -> Union[DataSet, Dict[str, DataSet]]:
        """
        从指定一个或多个路径中的文件中读取数据，返回一个或多个数据集 :class:`~fastNLP.DataSet` 。
        如果处理多个路径，传入的 dict 中的 key 与返回的 dict 中的 key 保存一致。

        :param Union[str, Dict[str, str]] paths: 文件路径
        :return: :class:`~fastNLP.DataSet` 类的对象或存储多个 :class:`~fastNLP.DataSet` 的字典
        """
        if isinstance(paths, str):
            return self._load(paths)
        return {name: self._load(path) for name, path in paths.items()}
    
    def _load(self, path: str) -> DataSet:
        """从指定路径的文件中读取数据,返回 :class:`~fastNLP.DataSet` 类型的对象

        :param str path: 文件路径
        :return: 一个 :class:`~fastNLP.DataSet` 类型的对象
        """
        raise NotImplementedError
    
    def process(self, paths: Union[str, Dict[str, str]], **options) -> DataInfo:
        """
        对于特定的任务和数据集，读取并处理数据，返回处理DataInfo类对象或字典。
        
        从指定一个或多个路径中的文件中读取数据，DataInfo对象中可以包含一个或多个数据集 。
        如果处理多个路径，传入的 dict 的 key 与返回DataInfo中的 dict 中的 key 保存一致。

        返回的 :class:`DataInfo` 对象有如下属性：
        
        - vocabs: 由从数据集中获取的词表组成的字典，每个词表
        - embeddings: (可选) 数据集对应的词嵌入
        - datasets: 一个dict，包含一系列 :class:`~fastNLP.DataSet` 类型的对象。其中 field 的命名参考 :mod:`~fastNLP.core.const`

        :param paths: 原始数据读取的路径
        :param options: 根据不同的任务和数据集，设计自己的参数
        :return: 返回一个 DataInfo
        """
        raise NotImplementedError


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
            'sentence1_parse': 'words1',
            'sentence2_parse': 'words2',
            'gold_label': 'target',
        }
        super(SNLILoader, self).__init__(fields=fields)
    
    def _load(self, path):
        ds = super(SNLILoader, self)._load(path)
        
        def parse_tree(x):
            t = Tree.fromstring(x)
            return t.leaves()
        
        ds.apply(lambda ins: parse_tree(
            ins['words1']), new_field_name='words1')
        ds.apply(lambda ins: parse_tree(
            ins['words2']), new_field_name='words2')
        ds.drop(lambda x: x['target'] == '-')
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

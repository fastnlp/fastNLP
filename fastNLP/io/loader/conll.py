__all__ = [
    "ConllLoader",
    "Conll2003Loader",
    "Conll2003NERLoader",
    "OntoNotesNERLoader",
    "CTBLoader",
    "CNNERLoader",
    "MsraNERLoader",
    "WeiboNERLoader",
    "PeopleDailyNERLoader"
]

import glob
import os
import random
import shutil
import time
from typing import List

from .loader import Loader
from ..file_reader import _read_conll
# from ...core.const import Const
from fastNLP.core.dataset import DataSet, Instance


class ConllLoader(Loader):
    r"""
    :class:`ConllLoader` 支持读取的数据格式：以空行隔开两个 sample，除了分割行之外的每一行用空格或者制表符隔开不同的元素。如下例所示::

        # 文件中的内容
        Nadim NNP B-NP B-PER
        Ladki NNP I-NP I-PER

        AL-AIN NNP B-NP B-LOC
        United NNP B-NP B-LOC
        Arab NNP I-NP I-LOC
        Emirates NNPS I-NP I-LOC
        1996-12-06 CD I-NP O
        ...

        # 如果用以下的参数读取，返回的DataSet将包含raw_words和pos两个field, 这两个field的值分别取自于第0列与第1列
        dataset = ConllLoader(headers=['raw_words', 'pos'], indexes=[0, 1])._load('/path/to/train.conll')
        # 如果用以下的参数读取，返回的DataSet将包含raw_words和ner两个field, 这两个field的值分别取自于第0列与第2列
        dataset = ConllLoader(headers=['raw_words', 'ner'], indexes=[0, 3])._load('/path/to/train.conll')
        # 如果用以下的参数读取，返回的DataSet将包含raw_words, pos和ner三个field
        dataset = ConllLoader(headers=['raw_words', 'pos', 'ner'], indexes=[0, 1, 3])._load('/path/to/train.conll')

    :class:`ConllLoader` 返回的 :class:`~fastNLP.core.DataSet` 的 `field` 由传入的 ``headers`` 确定。

    :param headers: 每一列数据的名称， ``header`` 与 ``indexes`` 一一对应
    :param sep: 指定分隔符，默认为制表符
    :param indexes: 需要保留的数据列下标，从 **0** 开始。若为 ``None`` ，则所有列都保留。
    :param dropna: 是否忽略非法数据，若为 ``False`` ，则遇到非法数据时抛出 :class:`ValueError` 。
    :param drophashtag: 是否忽略以 ``#`` 开头的句子。
    """
    
    def __init__(self, headers: List[str], sep: str=None, indexes: List[int]=None, dropna: bool=True, drophash: bool=True):
        super(ConllLoader, self).__init__()
        if not isinstance(headers, (list, tuple)):
            raise TypeError(
                'invalid headers: {}, should be list of strings'.format(headers))
        self.headers = headers
        self.dropna = dropna
        self.drophash = drophash
        self.sep=sep
        if indexes is None:
            self.indexes = list(range(len(self.headers)))
        else:
            if len(indexes) != len(headers):
                raise ValueError
            self.indexes = indexes
    
    def _load(self, path):
        r"""
        传入的一个文件路径，将该文件读入DataSet中，field由ConllLoader初始化时指定的headers决定。

        :param str path: 文件的路径
        :return: DataSet
        """
        ds = DataSet()
        for idx, data in _read_conll(path,sep=self.sep, indexes=self.indexes, dropna=self.dropna,
                                     drophash=self.drophash):
            ins = {h: data[i] for i, h in enumerate(self.headers)}
            ds.append(Instance(**ins))
        return ds


class Conll2003Loader(ConllLoader):
    r"""
    用于读取 **conll2003** 任务的数据。数据的内容应该类似于以下的内容：第一列为 **raw_words** ，第二列为 **pos** ，
    第三列为 **chunking** ，第四列为 **ner** 。
    数据中以 ``"-DOCSTART-"`` 开头的行将被忽略，因为该符号在 **conll2003** 中被用为文档分割符。

    Example::

        Nadim NNP B-NP B-PER
        Ladki NNP I-NP I-PER

        AL-AIN NNP B-NP B-LOC
        United NNP B-NP B-LOC
        Arab NNP I-NP I-LOC
        Emirates NNPS I-NP I-LOC
        1996-12-06 CD I-NP O
        ...

    读取的 :class:`~fastNLP.core.DataSet` 将具备以下的数据结构：

    .. csv-table:: 下面是 Conll2003Loader 加载后数据具备的结构。
       :header: "raw_words", "pos", "chunk", "ner"

       "[Nadim, Ladki]", "[NNP, NNP]", "[B-NP, I-NP]", "[B-PER, I-PER]"
       "[AL-AIN, United, Arab, ...]", "[NNP, NNP, NNP, ...]", "[B-NP, B-NP, I-NP, ...]", "[B-LOC, B-LOC, I-LOC, ...]"
       "[...]", "[...]", "[...]", "[...]"

    """
    
    def __init__(self):
        headers = [
            'raw_words', 'pos', 'chunk', 'ner',
        ]
        super(Conll2003Loader, self).__init__(headers=headers)
    
    def _load(self, path):
        r"""
        传入的一个文件路径，将该文件读入DataSet中，field由ConllLoader初始化时指定的headers决定。

        :param str path: 文件的路径
        :return: DataSet
        """
        ds = DataSet()
        for idx, data in _read_conll(path, indexes=self.indexes, dropna=self.dropna):
            doc_start = False
            for i, h in enumerate(self.headers):
                field = data[i]
                if str(field[0]).startswith('-DOCSTART-'):
                    doc_start = True
                    break
            if doc_start:
                continue
            ins = {h: data[i] for i, h in enumerate(self.headers)}
            ds.append(Instance(**ins))
        return ds
    
    def download(self, output_dir=None):
        raise RuntimeError("conll2003 cannot be downloaded automatically.")


class Conll2003NERLoader(ConllLoader):
    r"""
    用于读取 **conll2003** 任务的 NER 数据。每一行有 4 列内容，空行意味着隔开两个句子。

    支持读取的内容如下::

        Nadim NNP B-NP B-PER
        Ladki NNP I-NP I-PER

        AL-AIN NNP B-NP B-LOC
        United NNP B-NP B-LOC
        Arab NNP I-NP I-LOC
        Emirates NNPS I-NP I-LOC
        1996-12-06 CD I-NP O
        ...

    读取的 :class:`~fastNLP.core.DataSet` 将具备以下的数据结构：

    .. csv-table:: 下面是 Conll2003Loader 加载后数据具备的结构, target 是 BIO2 编码
       :header: "raw_words", "target"

       "[Nadim, Ladki]", "[B-PER, I-PER]"
       "[AL-AIN, United, Arab, ...]", "[B-LOC, B-LOC, I-LOC, ...]"
       "[...]",  "[...]"

    """
    
    def __init__(self):
        headers = [
            'raw_words', 'target',
        ]
        super().__init__(headers=headers, indexes=[0, 3])
    
    def _load(self, path):
        r"""
        传入的一个文件路径，将该文件读入DataSet中，field由ConllLoader初始化时指定的headers决定。

        :param str path: 文件的路径
        :return: DataSet
        """
        ds = DataSet()
        for idx, data in _read_conll(path, indexes=self.indexes, dropna=self.dropna):
            doc_start = False
            for i, h in enumerate(self.headers):
                field = data[i]
                if str(field[0]).startswith('-DOCSTART-'):
                    doc_start = True
                    break
            if doc_start:
                continue
            ins = {h: data[i] for i, h in enumerate(self.headers)}
            ds.append(Instance(**ins))
        if len(ds) == 0:
            raise RuntimeError("No data found {}.".format(path))
        return ds
    
    def download(self):
        raise RuntimeError("conll2003 cannot be downloaded automatically.")


class OntoNotesNERLoader(ConllLoader):
    r"""
    用以读取 **OntoNotes** 的 NER 数据，同时也是 **Conll2012** 的 NER 任务数据。将 **OntoNote** 数据处理为 conll 格式的过程可以参考
    https://github.com/yhcc/OntoNotes-5.0-NER。:class:`OntoNotesNERLoader` 将取第 **4** 列和第 **11** 列的内容。

    读取的数据格式为::

        bc/msnbc/00/msnbc_0000   0   0          Hi   UH   (TOP(FRAG(INTJ*)  -   -   -    Dan_Abrams  *   -
        bc/msnbc/00/msnbc_0000   0   1    everyone   NN              (NP*)  -   -   -    Dan_Abrams  *   -
        ...

    读取的 :class:`~fastNLP.core.DataSet` 将具备以下的数据结构：

    .. csv-table::
        :header: "raw_words", "target"

        "['Hi', 'everyone', '.']", "['O', 'O', 'O']"
        "['first', 'up', 'on', 'the', 'docket']", "['O', 'O', 'O', 'O', 'O']"
        "[...]", "[...]"

    """
    
    def __init__(self):
        super().__init__(headers=['raw_words', 'target'], indexes=[3, 10])
    
    def _load(self, path: str):
        dataset = super()._load(path)
        
        def convert_to_bio(tags):
            bio_tags = []
            flag = None
            for tag in tags:
                label = tag.strip("()*")
                if '(' in tag:
                    bio_label = 'B-' + label
                    flag = label
                elif flag:
                    bio_label = 'I-' + flag
                else:
                    bio_label = 'O'
                if ')' in tag:
                    flag = None
                bio_tags.append(bio_label)
            return bio_tags
        
        def convert_word(words):
            converted_words = []
            for word in words:
                word = word.replace('/.', '.')  # 有些结尾的.是/.形式的
                if not word.startswith('-'):
                    converted_words.append(word)
                    continue
                # 以下是由于这些符号被转义了，再转回来
                tfrs = {'-LRB-': '(',
                        '-RRB-': ')',
                        '-LSB-': '[',
                        '-RSB-': ']',
                        '-LCB-': '{',
                        '-RCB-': '}'
                        }
                if word in tfrs:
                    converted_words.append(tfrs[word])
                else:
                    converted_words.append(word)
            return converted_words
        
        dataset.apply_field(convert_word, field_name='raw_words', new_field_name='raw_words')
        dataset.apply_field(convert_to_bio, field_name='target', new_field_name='target')
        
        return dataset
    
    def download(self):
        raise RuntimeError("Ontonotes cannot be downloaded automatically, you can refer "
                           "https://github.com/yhcc/OntoNotes-5.0-NER to download and preprocess.")


class CTBLoader(Loader):
    r"""
    **CTB** 数据集的 **Loader**。支持加载的数据应该具备以下格式, 其中第二列为 **词语** ，第四列为 **pos tag** ，第七列为 **依赖树的 head** ，
    第八列为 **依赖树的 label** 。

    Example::

        1       印度    _       NR      NR      _       3       nn      _       _
        2       海军    _       NN      NN      _       3       nn      _       _
        3       参谋长  _       NN      NN      _       5       nsubjpass       _       _
        4       被      _       SB      SB      _       5       pass    _       _
        5       解职    _       VV      VV      _       0       root    _       _

        1       新华社  _       NR      NR      _       7       dep     _       _
        2       新德里  _       NR      NR      _       7       dep     _       _
        3       １２月  _       NT      NT      _       7       dep     _       _
        ...

    读取的 :class:`~fastNLP.core.DataSet` 将具备以下的数据结构：

    .. csv-table::
        :header: "raw_words", "pos", "dep_head", "dep_label"

        "[印度, 海军, ...]", "[NR, NN, SB, ...]", "[3, 3, ...]", "[nn, nn, ...]"
        "[新华社, 新德里, ...]", "[NR, NR, NT, ...]", "[7, 7, 7, ...]", "[dep, dep, dep, ...]"
        "[...]", "[...]", "[...]", "[...]"

    """
    def __init__(self):
        super().__init__()
        headers = [
            'raw_words', 'pos', 'dep_head', 'dep_label',
        ]
        indexes = [
            1, 3, 6, 7,
        ]
        self.loader = ConllLoader(headers=headers, indexes=indexes)
    
    def _load(self, path: str):
        dataset = self.loader._load(path)
        return dataset

    def download(self):
        r"""
        由于版权限制，不能提供自动下载功能。可参考

        https://catalog.ldc.upenn.edu/LDC2013T21
        """
        raise RuntimeError("CTB cannot be downloaded automatically.")


class CNNERLoader(Loader):
    r"""
    支持加载形如以下格式的内容，一行两列，以空格隔开两个 sample

    Example::

        我 O
        们 O
        变 O
        而 O
        以 O
        书 O
        会 O
        ...

    """
    def _load(self, path: str):
        """
        :param path: 文件路径
        :return: :class:`~fastNLP.core.DataSet` ，包含 ``raw_words`` 列和 ``target`` 列
        """
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            raw_chars = []
            target = []
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split()
                    if len(parts) == 1:  # 网上下载的数据有一些列少tag，默认补充O
                        parts.append('O')
                    raw_chars.append(parts[0])
                    target.append(parts[1])
                else:
                    if raw_chars:
                        ds.append(Instance(raw_chars=raw_chars, target=target))
                    raw_chars = []
                    target = []
        return ds


class MsraNERLoader(CNNERLoader):
    r"""
    读取 **MSRA-NER** 数据，如果您要使用该数据，请引用以下的文章：
    
    Gina-Anne Levow, 2006, The Third International Chinese Language Processing Bakeoff: Word Segmentation and Named Entity Recognition.
        
    数据中的格式应该类似于下列的内容::

        把	O
        欧	B-LOC

        美	B-LOC
        、	O

        港	B-LOC
        台	B-LOC

        流	O
        行	O

        的	O

        食	O

        ...

    读取的 :class:`~fastNLP.core.DataSet` 将具备以下的数据结构：

    .. csv-table::
        :header: "raw_chars", "target"

        "['把', '欧'] ", "['O', 'B-LOC']"
        "['美', '、']", "['B-LOC', 'O']"
        "[...]", "[...]"

    """
    
    def __init__(self):
        super().__init__()
    
    def download(self, dev_ratio: float = 0.1, re_download: bool = False) -> str:
        r"""
        自动下载 **MSAR-NER** 的数据。

        下载完成后在 ``output_dir`` 中有 ``train.conll`` , ``test.conll`` , ``dev.conll`` 三个文件。
        如果 ``dev_ratio`` 为 0，则只有 ``train.conll`` 和 ``test.conll`` 。

        :param dev_ratio: 如果路径中没有验证集 ，从 train 划分多少作为 dev 的数据。如果为 **0** ，则不划分 dev
        :param re_download: 是否重新下载数据，以重新切分数据。
        :return: 数据集的目录地址
        :return:
        """
        dataset_name = 'msra-ner'
        data_dir = self._get_dataset_path(dataset_name=dataset_name)
        modify_time = 0
        for filepath in glob.glob(os.path.join(data_dir, '*')):
            modify_time = os.stat(filepath).st_mtime
            break
        if time.time() - modify_time > 1 and re_download:  # 通过这种比较丑陋的方式判断一下文件是否是才下载的
            shutil.rmtree(data_dir)
            data_dir = self._get_dataset_path(dataset_name=dataset_name)
        
        if not os.path.exists(os.path.join(data_dir, 'dev.conll')):
            if dev_ratio > 0:
                assert 0 < dev_ratio < 1, "dev_ratio should be in range (0,1)."
                try:
                    with open(os.path.join(data_dir, 'train.conll'), 'r', encoding='utf-8') as f, \
                            open(os.path.join(data_dir, 'middle_file.conll'), 'w', encoding='utf-8') as f1, \
                            open(os.path.join(data_dir, 'dev.conll'), 'w', encoding='utf-8') as f2:
                        lines = []  # 一个sample包含很多行
                        for line in f:
                            line = line.strip()
                            if line:
                                lines.append(line)
                            else:
                                if random.random() < dev_ratio:
                                    f2.write('\n'.join(lines) + '\n\n')
                                else:
                                    f1.write('\n'.join(lines) + '\n\n')
                                lines.clear()
                    os.remove(os.path.join(data_dir, 'train.conll'))
                    os.renames(os.path.join(data_dir, 'middle_file.conll'), os.path.join(data_dir, 'train.conll'))
                finally:
                    if os.path.exists(os.path.join(data_dir, 'middle_file.conll')):
                        os.remove(os.path.join(data_dir, 'middle_file.conll'))
        
        return data_dir


class WeiboNERLoader(CNNERLoader):
    r"""
    读取 **WeiboNER** 数据，如果您要使用该数据，请引用以下的文章：
    
    Nanyun Peng and Mark Dredze, 2015, Named Entity Recognition for Chinese Social Media with Jointly Trained Embeddings.
    
    数据中的格式应该类似与下列的内容::

        老	B-PER.NOM
        百	I-PER.NOM
        姓	I-PER.NOM

        心	O

        ...

    读取的 :class:`~fastNLP.core.DataSet` 将具备以下的数据结构：

        .. csv-table::

            :header: "raw_chars", "target"

            "['老', '百', '姓']", "['B-PER.NOM', 'I-PER.NOM', 'I-PER.NOM']"
            "['心']", "['O']"
            "[...]", "[...]"

        """
    def __init__(self):
        super().__init__()
    
    def download(self) -> str:
        r"""
        自动下载 **Weibo-NER** 的数据。

        :return: 数据集目录地址
        """
        dataset_name = 'weibo-ner'
        data_dir = self._get_dataset_path(dataset_name=dataset_name)
        
        return data_dir


class PeopleDailyNERLoader(CNNERLoader):
    r"""
    加载 **People's Daily NER** 数据集的 **Loader** 。支持加载的数据格式如下::

        中 B-ORG
        共 I-ORG
        中 I-ORG
        央 I-ORG

        致 O
        中 B-ORG
        ...

    读取的 :class:`~fastNLP.core.DataSet` 将具备以下的数据结构：

    .. csv-table:: target 列是基于 BIO 的编码方式
        :header: "raw_chars", "target"

        "['中', '共', '中', '央']", "['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG']"
        "[...]", "[...]"

    """
    
    def __init__(self):
        super().__init__()
    
    def download(self) -> str:
        """
        自动下载数据集。

        :return: 数据集目录地址
        """
        dataset_name = 'peopledaily'
        data_dir = self._get_dataset_path(dataset_name=dataset_name)
        
        return data_dir

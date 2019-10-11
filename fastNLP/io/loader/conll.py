"""undocumented"""

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

from .loader import Loader
from ..file_reader import _read_conll
from ...core.const import Const
from ...core.dataset import DataSet
from ...core.instance import Instance


class ConllLoader(Loader):
    """
    ConllLoader支持读取的数据格式: 以空行隔开两个sample，除了分割行，每一行用空格或者制表符隔开不同的元素。如下例所示:

    Example::

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

    ConllLoader返回的DataSet的field由传入的headers确定。

    数据中以"-DOCSTART-"开头的行将被忽略，因为该符号在conll 2003中被用为文档分割符。

    """
    
    def __init__(self, headers, indexes=None, dropna=True):
        """
        
        :param list headers: 每一列数据的名称，需为List or Tuple  of str。``header`` 与 ``indexes`` 一一对应
        :param list indexes: 需要保留的数据列下标，从0开始。若为 ``None`` ，则所有列都保留。Default: ``None``
        :param bool dropna: 是否忽略非法数据，若 ``False`` ，遇到非法数据时抛出 ``ValueError`` 。Default: ``True``
        """
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
        """
        传入的一个文件路径，将该文件读入DataSet中，field由ConllLoader初始化时指定的headers决定。

        :param str path: 文件的路径
        :return: DataSet
        """
        ds = DataSet()
        for idx, data in _read_conll(path, indexes=self.indexes, dropna=self.dropna):
            ins = {h: data[i] for i, h in enumerate(self.headers)}
            ds.append(Instance(**ins))
        return ds


class Conll2003Loader(ConllLoader):
    """
    用于读取conll2003任务的数据。数据的内容应该类似与以下的内容, 第一列为raw_words, 第二列为pos, 第三列为chunking，第四列为ner。

    Example::

        Nadim NNP B-NP B-PER
        Ladki NNP I-NP I-PER

        AL-AIN NNP B-NP B-LOC
        United NNP B-NP B-LOC
        Arab NNP I-NP I-LOC
        Emirates NNPS I-NP I-LOC
        1996-12-06 CD I-NP O
        ...

    返回的DataSet的内容为

    .. csv-table:: 下面是Conll2003Loader加载后数据具备的结构。
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
        """
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
    """
    用于读取conll2003任务的NER数据。每一行有4列内容，空行意味着隔开两个句子

    支持读取的内容如下
    Example::

        Nadim NNP B-NP B-PER
        Ladki NNP I-NP I-PER

        AL-AIN NNP B-NP B-LOC
        United NNP B-NP B-LOC
        Arab NNP I-NP I-LOC
        Emirates NNPS I-NP I-LOC
        1996-12-06 CD I-NP O
        ...

    返回的DataSet的内容为

    .. csv-table:: 下面是Conll2003Loader加载后数据具备的结构, target是BIO2编码
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
        """
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
    """
    用以读取OntoNotes的NER数据，同时也是Conll2012的NER任务数据。将OntoNote数据处理为conll格式的过程可以参考
    https://github.com/yhcc/OntoNotes-5.0-NER。OntoNoteNERLoader将取第4列和第11列的内容。

    读取的数据格式为：

    Example::

        bc/msnbc/00/msnbc_0000   0   0          Hi   UH   (TOP(FRAG(INTJ*)  -   -   -    Dan_Abrams  *   -
        bc/msnbc/00/msnbc_0000   0   1    everyone   NN              (NP*)  -   -   -    Dan_Abrams  *   -
        ...

    返回的DataSet的内容为

    .. csv-table::
        :header: "raw_words", "target"

        "['Hi', 'everyone', '.']", "['O', 'O', 'O']"
        "['first', 'up', 'on', 'the', 'docket'], "['O', 'O', 'O', 'O', 'O']"
        "[...]", "[...]"

    """
    
    def __init__(self):
        super().__init__(headers=[Const.RAW_WORD, Const.TARGET], indexes=[3, 10])
    
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
        
        dataset.apply_field(convert_word, field_name=Const.RAW_WORD, new_field_name=Const.RAW_WORD)
        dataset.apply_field(convert_to_bio, field_name=Const.TARGET, new_field_name=Const.TARGET)
        
        return dataset
    
    def download(self):
        raise RuntimeError("Ontonotes cannot be downloaded automatically, you can refer "
                           "https://github.com/yhcc/OntoNotes-5.0-NER to download and preprocess.")


class CTBLoader(Loader):
    """
    支持加载的数据应该具备以下格式, 其中第二列为词语，第四列为pos tag，第七列为依赖树的head，第八列为依赖树的label

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

    读取之后DataSet具备的格式为

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
        """
        由于版权限制，不能提供自动下载功能。可参考

        https://catalog.ldc.upenn.edu/LDC2013T21

        :return:
        """
        raise RuntimeError("CTB cannot be downloaded automatically.")


class CNNERLoader(Loader):
    def _load(self, path: str):
        """
        支持加载形如以下格式的内容，一行两列，以空格隔开两个sample

        Example::

            我 O
            们 O
            变 O
            而 O
            以 O
            书 O
            会 O
            ...

        :param str path: 文件路径
        :return: DataSet，包含raw_words列和target列
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
    """
    读取MSRA-NER数据，数据中的格式应该类似与下列的内容

    Example::

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

    读取后的DataSet包含以下的field

    .. csv-table::
        :header: "raw_chars", "target"

        "['把', '欧'] ", "['O', 'B-LOC']"
        "['美', '、']", "['B-LOC', 'O']"
        "[...]", "[...]"

    """
    
    def __init__(self):
        super().__init__()
    
    def download(self, dev_ratio: float = 0.1, re_download: bool = False) -> str:
        """
        自动下载MSAR-NER的数据，如果你使用该数据，请引用 Gina-Anne Levow, 2006, The Third International Chinese Language
        Processing Bakeoff: Word Segmentation and Named Entity Recognition.

        根据dev_ratio的值随机将train中的数据取出一部分作为dev数据。下载完成后在output_dir中有train.conll, test.conll,
        dev.conll三个文件。

        :param float dev_ratio: 如果路径中没有dev集，从train划分多少作为dev的数据. 如果为0，则不划分dev。
        :param bool re_download: 是否重新下载数据，以重新切分数据。
        :return: str, 数据集的目录地址
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
    """
    读取WeiboNER数据，数据中的格式应该类似与下列的内容

    Example::

        老	B-PER.NOM
        百	I-PER.NOM
        姓	I-PER.NOM

        心	O

        ...

        读取后的DataSet包含以下的field

        .. csv-table::

            :header: "raw_chars", "target"

            "['老', '百', '姓']", "['B-PER.NOM', 'I-PER.NOM', 'I-PER.NOM']"
            "['心']", "['O']"
            "[...]", "[...]"

        """
    def __init__(self):
        super().__init__()
    
    def download(self) -> str:
        """
        自动下载Weibo-NER的数据，如果你使用了该数据，请引用 Nanyun Peng and Mark Dredze, 2015, Named Entity Recognition for
        Chinese Social Media with Jointly Trained Embeddings.

        :return: str
        """
        dataset_name = 'weibo-ner'
        data_dir = self._get_dataset_path(dataset_name=dataset_name)
        
        return data_dir


class PeopleDailyNERLoader(CNNERLoader):
    """
    支持加载的数据格式如下

    Example::

        中 B-ORG
        共 I-ORG
        中 I-ORG
        央 I-ORG

        致 O
        中 B-ORG
        ...

    读取后的DataSet包含以下的field

    .. csv-table:: target列是基于BIO的编码方式
        :header: "raw_chars", "target"

        "['中', '共', '中', '央']", "['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG']"
        "[...]", "[...]"

    """
    
    def __init__(self):
        super().__init__()
    
    def download(self) -> str:
        dataset_name = 'peopledaily'
        data_dir = self._get_dataset_path(dataset_name=dataset_name)
        
        return data_dir

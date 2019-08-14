from typing import Dict, Union

from .loader import Loader
from ...core.dataset import DataSet
from ..file_reader import _read_conll
from ...core.instance import Instance
from .. import DataBundle
from ..utils import check_loader_paths
from ...core.const import Const


class ConllLoader(Loader):
    """
    别名：:class:`fastNLP.io.ConllLoader` :class:`fastNLP.io.data_loader.ConllLoader`

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

    :param list headers: 每一列数据的名称，需为List or Tuple  of str。``header`` 与 ``indexes`` 一一对应
    :param list indexes: 需要保留的数据列下标，从0开始。若为 ``None`` ，则所有列都保留。Default: ``None``
    :param bool dropna: 是否忽略非法数据，若 ``False`` ，遇到非法数据时抛出 ``ValueError`` 。Default: ``True``

    """
    def __init__(self, headers, indexes=None, dropna=True):
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
    用于读取conll2003任务的NER数据。

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
        return ds

    def download(self):
        raise RuntimeError("conll2003 cannot be downloaded automatically.")


class OntoNotesNERLoader(ConllLoader):
    """
    用以读取OntoNotes的NER数据，同时也是Conll2012的NER任务数据。将OntoNote数据处理为conll格式的过程可以参考
    https://github.com/yhcc/OntoNotes-5.0-NER。OntoNoteNERLoader将取第4列和第11列的内容。

    返回的DataSet的内容为

    .. csv-table:: 下面是使用OntoNoteNERLoader读取的DataSet所具备的结构, target列是BIO编码
        :header: "raw_words", "target"

        "[Nadim, Ladki]", "[B-PER, I-PER]"
        "[AL-AIN, United, Arab, ...]", "[B-LOC, B-LOC, I-LOC, ...]"
        "[...]", "[...]"

    """

    def __init__(self):
        super().__init__(headers=[Const.RAW_WORD, Const.TARGET], indexes=[3, 10])

    def _load(self, path:str):
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
                tfrs = {'-LRB-':'(',
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
    def __init__(self):
        super().__init__()

    def _load(self, path:str):
        pass

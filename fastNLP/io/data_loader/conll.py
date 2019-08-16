
from ...core.dataset import DataSet
from ...core.instance import Instance
from ..data_bundle import DataSetLoader
from ..file_reader import _read_conll
from typing import Union, Dict
from ..utils import check_loader_paths
from ..data_bundle import DataBundle

class ConllLoader(DataSetLoader):
    """
    别名：:class:`fastNLP.io.ConllLoader` :class:`fastNLP.io.data_loader.ConllLoader`

    该ConllLoader支持读取的数据格式: 以空行隔开两个sample，除了分割行，每一行用空格或者制表符隔开不同的元素。如下例所示:

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

    dataset = ConllLoader(headers=['raw_words', 'pos'], indexes=[0, 1])._load('/path/to/train.conll')中DataSet的raw_words
    列与pos列的内容都是List[str]

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
        传入的一个文件路径，将该文件读入DataSet中，field由Loader初始化时指定的headers决定。

        :param str path: 文件的路径
        :return: DataSet
        """
        ds = DataSet()
        for idx, data in _read_conll(path, indexes=self.indexes, dropna=self.dropna):
            ins = {h: data[i] for i, h in enumerate(self.headers)}
            ds.append(Instance(**ins))
        return ds

    def load(self, paths: Union[str, Dict[str, str]]) -> DataBundle:
        """
        从指定一个或多个路径中的文件中读取数据，返回:class:`~fastNLP.io.DataBundle` 。

        读取的field根据ConllLoader初始化时传入的headers决定。

        :param Union[str, Dict[str, str]] paths: 支持以下的几种输入方式
            (1) 传入一个目录, 该目录下名称包含train的被认为是train，包含test的被认为是test，包含dev的被认为是dev，如果检测到多个文件
                名包含'train'、 'dev'、 'test'则会报错

                Example::
                    data_bundle = ConllLoader().load('/path/to/dir')  # 返回的DataBundle中datasets根据目录下是否检测到train, dev, test等有所变化
                    # 可以通过以下的方式取出DataSet
                    tr_data = data_bundle.datasets['train']
                    te_data = data_bundle.datasets['test']  # 如果目录下有文件包含test这个字段

            (2) 传入文件path

                Example::
                    data_bundle = ConllLoader().load("/path/to/a/train.conll") # 返回DataBundle对象, datasets中仅包含'train'
                    tr_data = data_bundle.datasets['train']  # 可以通过以下的方式取出DataSet

            (3) 传入一个dict，比如train，dev，test不在同一个目录下，或者名称中不包含train, dev, test

                Example::
                    paths = {'train':"/path/to/tr.conll", 'dev':"/to/validate.conll", "test":"/to/te.conll"}
                    data_bundle = ConllLoader().load(paths)  # 返回的DataBundle中的dataset中包含"train", "dev", "test"
                    dev_data = data_bundle.datasets['dev']

        :return: :class:`~fastNLP.DataSet` 类的对象或 :class:`~fastNLP.io.DataBundle` 的字典
        """
        paths = check_loader_paths(paths)
        datasets = {name: self._load(path) for name, path in paths.items()}
        data_bundle = DataBundle(datasets=datasets)
        return data_bundle


class Conll2003Loader(ConllLoader):
    """
    别名：:class:`fastNLP.io.Conll2003Loader` :class:`fastNLP.io.data_loader.Conll2003Loader`

    该Loader用以读取Conll2003数据，conll2003的数据可以在https://github.com/davidsbatista/NER-datasets/tree/master/CONLL2003
    找到。数据中以"-DOCSTART-"开头的行将被忽略，因为该符号在conll 2003中被用为文档分割符。

    返回的DataSet将具有以下['raw_words', 'pos', 'chunks', 'ner']四个field, 每个field中的内容都是List[str]。

    .. csv-table:: Conll2003Loader处理之       :header: "raw_words", "words", "target", "seq_len"

       "[Nadim, Ladki]", "[1, 2]", "[1, 2]", 2
       "[AL-AIN, United, Arab, ...]", "[3, 4, 5,...]", "[3, 4]", 5
       "[...]", "[...]", "[...]", .

    """

    def __init__(self):
        headers = [
            'raw_words', 'pos', 'chunks', 'ner',
        ]
        super(Conll2003Loader, self).__init__(headers=headers)

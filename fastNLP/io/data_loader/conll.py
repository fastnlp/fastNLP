
from ...core.dataset import DataSet
from ...core.instance import Instance
from ..base_loader import DataSetLoader
from ..file_reader import _read_conll


class ConllLoader(DataSetLoader):
    """
    别名：:class:`fastNLP.io.ConllLoader` :class:`fastNLP.io.data_loader.ConllLoader`

    读取Conll格式的数据. 数据格式详见 http://conll.cemantix.org/2012/data.html. 数据中以"-DOCSTART-"开头的行将被忽略，因为
    该符号在conll 2003中被用为文档分割符。

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

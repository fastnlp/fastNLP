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
]


from ..core.dataset import DataSet
from ..core.instance import Instance
from .file_reader import _read_csv, _read_json
from .base_loader import DataSetLoader


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

__all__ = [
    "JsonLoader"
]

from .loader import Loader
from ..file_reader import _read_json
from fastNLP.core.dataset import DataSet, Instance


class JsonLoader(Loader):
    r"""
    读取 *json* 格式数据，数据必须按行存储，每行是一个包含各类属性的 json 对象。

    :param fields: 需要读入的 json 属性名称，和读入后在 :class:`~fastNLP.core.DataSet` 中存储的 `field_name`。
        ``fields`` 的 `key` 必须是 json 对象的 **属性名**， ``fields`` 的 `value` 为读入后在 ``DataSet`` 存储的 `field_name` ，
        `value` 也可为 ``None`` ，这时读入后的  `field_name` 与 json 对象对应属性同名。
        ``fields`` 可为 ``None`` ，这时 json 对象所有属性都保存在 ``DataSet`` 中。
    :param dropna: 是否忽略非法数据，若为 ``True`` 则忽略；若为 ``False`` 则在遇到非法数据时抛出 :class:`ValueError`。
    """

    def __init__(self, fields: dict=None, dropna=False):
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

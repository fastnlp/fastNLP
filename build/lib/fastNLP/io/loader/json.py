r"""undocumented"""

__all__ = [
    "JsonLoader"
]

from .loader import Loader
from ..file_reader import _read_json
from ...core.dataset import DataSet
from ...core.instance import Instance


class JsonLoader(Loader):
    r"""
    别名：:class:`fastNLP.io.JsonLoader` :class:`fastNLP.io.loader.JsonLoader`

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

r"""
instance 模块实现了 Instance 类，即在 fastNLP 中 sample 对应的类型。一个 sample 可以认为是一个 Instance 类型的对象。
便于理解的例子可以参考文档 :mod:`fastNLP.core.dataset.dataset` 。
"""

__all__ = [
    "Instance"
]

from typing import Mapping
from fastNLP.core.utils.utils import pretty_table_printer


class Instance(Mapping):
    r"""
    Instance 是 fastNLP 中对应一个 sample 的类。每个 sample 在 fastNLP 中是一个 Instance 对象。
    Instance 一般与 :class:`~fastNLP.DataSet` 一起使用, Instance 的初始化如下面的代码所示::

        >>> instance = Instance(input="this is a demo sentence", label='good')

    """

    def __init__(self, **fields):

        self.fields = fields

    def add_field(self, field_name: str, field: any):
        r"""
        向 Instance 中增加一个 field

        :param field_name: 新增 field 的名称
        :param field: 新增 field 的内容
        """
        self.fields[field_name] = field

    def items(self):
        r"""
        返回一个迭代器，迭代器返回两个内容，第一个内容是 field_name, 第二个内容是 field_value

        :return: 一个迭代器
        """
        return self.fields.items()

    def keys(self):
        r"""
        返回一个迭代器，内容是 field_name

        :return: 一个迭代器
        """
        return self.fields.keys()

    def values(self):
        r"""
        返回一个迭代器，内容是 field_value

        :return: 一个迭代器
        """
        return self.fields.values()

    def __contains__(self, item):
        return item in self.fields

    def __getitem__(self, name):
        if name in self.fields:
            return self.fields[name]
        else:
            raise KeyError("{} not found".format(name))

    def __setitem__(self, name, field):
        return self.add_field(name, field)

    def __repr__(self):
        return str(pretty_table_printer(self))

    def __len__(self):
        return len(self.fields)

    def __iter__(self):
        return iter(self.fields)

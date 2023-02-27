from typing import Any, Mapping

from fastNLP.core.utils.utils import pretty_table_printer

__all__ = ['Instance']


class Instance(Mapping):
    r"""
    **fastNLP** 中 sample 对应的类型。

    每个 sample 在 **fastNLP** 中都可以认为是一个 ``Instance`` 对象，一般与
    :class:`.DataSet` 一起使用，``Instance`` 的初始化如下面的代码所示:

        >>> instance = Instance(input="this is a demo sentence", label='good')

    便于理解的例子可以参考 :class:`.DataSet` 中的样例。

    """

    def __init__(self, **fields):

        self.fields = fields

    def add_field(self, field_name: str, field: Any):
        r"""
        向 Instance 中增加一个 field

        :param field_name: 新增 field 的名称
        :param field: 新增 field 的内容
        """
        self.fields[field_name] = field

    def items(self):
        r"""
        返回一个迭代器，迭代器返回两个内容，第一个内容是 field_name, 第二个内容是
        field_value

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
            raise KeyError('{} not found'.format(name))

    def __setitem__(self, name, field):
        return self.add_field(name, field)

    def __repr__(self):
        return str(pretty_table_printer(self))

    def __len__(self):
        return len(self.fields)

    def __iter__(self):
        return iter(self.fields)

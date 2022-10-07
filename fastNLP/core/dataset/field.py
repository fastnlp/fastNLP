r"""
.. todo::
    doc
"""
__all__ = [
    'FieldArray'
]

from collections import Counter
from typing import Any, Union, List, Callable
from ..log import logger

import numpy as np


class FieldArray:
    """
    :class:`~fastNLP.core.dataset.DatSet` 中用于表示列的数据类型。

    :param name: 字符串的名称
    :param content: 任意类型的数据
    """

    def __init__(self, name: str, content):
        if len(content) == 0:
            raise RuntimeError("Empty fieldarray is not allowed.")
        _content = content
        try:
            _content = list(_content)
        except BaseException as e:
            logger.error(f"Cannot convert content(of type:{type(content)}) into list.")
            raise e
        self.name = name
        self.content = _content

    def append(self, val: Any) -> None:
        r"""
        :param val: 把该 ``val`` 添加到 fieldarray 中。
        """
        self.content.append(val)

    def pop(self, index: int) -> None:
        r"""
        删除该 field 中 ``index`` 处的元素

        :param index: 从 ``0`` 开始的数据下标。
        """
        self.content.pop(index)

    def __iter__(self):
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, indices: Union[int, List[int]]):
        return self.get(indices)

    def __setitem__(self, idx: int, val: Any):
        assert isinstance(idx, int)
        self.content[idx] = val

    def get(self, indices: Union[int, List[int]]):
        r"""
        根据给定的 ``indices`` 返回内容。

        :param indices: 获取 ``indices`` 对应的内容。
        :return: 根据给定的 ``indices`` 返回的内容，可能是单个值 或 :class:`numpy.ndarray`
        """
        if isinstance(indices, int):
            if indices == -1:
                indices = len(self) - 1
            assert 0 <= indices < len(self)
            return self.content[indices]
        try:
            contents = [self.content[i] for i in indices]
        except BaseException as e:
            raise e
        return np.array(contents)

    def __len__(self):
        r"""
        返回长度

        :return:
        """
        return len(self.content)

    def split(self, sep: str = None, inplace: bool = True):
        r"""
        依次对自身的元素使用 ``.split()`` 方法，应该只有当本 field 的元素为 :class:`str` 时，该方法才有用。

        :param sep: 分割符，如果为 ``None`` 则直接调用 ``str.split()``。
        :param inplace: 如果为 ``True``，则将新生成值替换本 field。否则返回 :class:`list`。
        :return: List[List[str]] or self
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                new_contents.append(cell.split(sep))
            except Exception as e:
                logger.error(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def int(self, inplace: bool = True):
        r"""
        将本 field 中的值调用 ``int(cell)``. 支持 field 中内容为以下两种情况:

            * ['1', '2', ...](即 field 中每个值为 :class:`str` 的)，
            * [['1', '2', ..], ['3', ..], ...](即 field 中每个值为一个 :class:`list` ，:class:`list` 中的值会被依次转换。)

        :param inplace: 如果为 ``True``，则将新生成值替换本 field，并返回当前 field 。否则返回 :class:`list`。
        :return: List[int], List[List[int]], self
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([int(value) for value in cell])
                else:
                    new_contents.append(int(cell))
            except Exception as e:
                print(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def float(self, inplace=True):
        r"""
        将本 field 中的值调用 ``float(cell)``. 支持 field 中内容为以下两种情况:

            * ['1', '2', ...](即 field 中每个值为 :class:`str` 的)，
            * [['1', '2', ..], ['3', ..], ...](即 field 中每个值为一个 :class:`list` ，:class:`list` 中的值会被依次转换。)

        :param inplace: 如果为 ``True``，则将新生成值替换本 field，并返回当前 field 。否则返回 :class:`list`。
        :return:
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([float(value) for value in cell])
                else:
                    new_contents.append(float(cell))
            except Exception as e:
                print(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def bool(self, inplace=True):
        r"""
        将本field中的值调用 ``bool(cell)``. 支持 field 中内容为以下两种情况

            * ['1', '2', ...](即 field 中每个值为 :class:`str` 的)，
            * [['1', '2', ..], ['3', ..], ...](即 field 中每个值为一个 :class:`list` ，:class:`list` 中的值会被依次转换。)

        :param inplace: 如果为 ``True``，则将新生成值替换本 field，并返回当前 field 。否则返回 :class:`list`。
        :return:
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([bool(value) for value in cell])
                else:
                    new_contents.append(bool(cell))
            except Exception as e:
                print(f"Exception happens when process value in index {index}.")
                raise e

        return self._after_process(new_contents, inplace=inplace)

    def lower(self, inplace=True):
        r"""
        将本 field 中的值调用 ``cell.lower()``， 支持 field 中内容为以下两种情况

            * ['1', '2', ...](即 field 中每个值为 :class:`str` 的)，
            * [['1', '2', ..], ['3', ..], ...](即 field 中每个值为一个 :class:`list` ，:class:`list` 中的值会被依次转换。)

        :param inplace: 如果为 ``True``，则将新生成值替换本 field，并返回当前 field 。否则返回 :class:`list`。
        :return: List[int], List[List[int]], self
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([value.lower() for value in cell])
                else:
                    new_contents.append(cell.lower())
            except Exception as e:
                print(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def upper(self, inplace=True):
        r"""
        将本 field 中的值调用 ``cell.upper()``， 支持 field 中内容为以下两种情况

            * ['1', '2', ...](即 field 中每个值为 :class:`str` 的)，
            * [['1', '2', ..], ['3', ..], ...](即 field 中每个值为一个 :class:`list` ，:class:`list` 中的值会被依次转换。)

        :param inplace: 如果为 ``True``，则将新生成值替换本 field，并返回当前 field 。否则返回 :class:`list`。
        :return: List[int], List[List[int]], self
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                if isinstance(cell, list):
                    new_contents.append([value.upper() for value in cell])
                else:
                    new_contents.append(cell.upper())
            except Exception as e:
                print(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def value_count(self) -> Counter:
        r"""
        返回该 field 下不同 value 的数量。多用于统计 label 数量

        :return: 计数结果，key 是 label，value 是出现次数
        """
        count = Counter()

        def cum(cells):
            if isinstance(cells, Callable) and not isinstance(cells, str):
                for cell_ in cells:
                    cum(cell_)
            else:
                count[cells] += 1

        for cell in self.content:
            cum(cell)
        return count

    def _after_process(self, new_contents: list, inplace: bool):
        r"""
        当调用处理函数之后，决定是否要替换 field。

        :param new_contents:
        :param inplace:
        :return: self或者生成的content
        """
        if inplace:
            self.content = new_contents
            return self
        else:
            return new_contents

r"""
.. todo::
    doc
"""
__all__ = [
    'FieldArray'
]

from collections import Counter
from typing import Any, Union, List, Callable

import numpy as np


class FieldArray:

    def __init__(self, name: str, content):
        if len(content) == 0:
            raise RuntimeError("Empty fieldarray is not allowed.")
        _content = content
        try:
            _content = list(_content)
        except BaseException as e:
            print(f"Cannot convert content(of type:{type(content)}) into list.")
            raise e
        self.name = name
        self.content = _content

    def append(self, val: Any) -> None:
        r"""
        :param val: 把该val append到fieldarray。
        :return:
        """
        self.content.append(val)

    def pop(self, index: int) -> None:
        r"""
        删除该field中index处的元素
        :param int index: 从0开始的数据下标。
        :return:
        """
        self.content.pop(index)

    def __getitem__(self, indices: Union[int, List[int]]):
        return self.get(indices)

    def __setitem__(self, idx: int, val: Any):
        assert isinstance(idx, int)
        self.content[idx] = val

    def get(self, indices: Union[int, List[int]]):
        r"""
        根据给定的indices返回内容。

        :param int,List[int] indices: 获取indices对应的内容。
        :return: 根据给定的indices返回的内容，可能是单个值或ndarray
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
        Returns the size of FieldArray.

        :return int length:
        """
        return len(self.content)

    def split(self, sep: str = None, inplace: bool = True):
        r"""
        依次对自身的元素使用.split()方法，应该只有当本field的元素为str时，该方法才有用。

        :param sep: 分割符，如果为None则直接调用str.split()。
        :param inplace: 如果为True，则将新生成值替换本field。否则返回list。
        :return: List[List[str]] or self
        """
        new_contents = []
        for index, cell in enumerate(self.content):
            try:
                new_contents.append(cell.split(sep))
            except Exception as e:
                print(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)

    def int(self, inplace: bool = True):
        r"""
        将本field中的值调用int(cell). 支持field中内容为以下两种情况(1)['1', '2', ...](即field中每个值为str的)，
            (2) [['1', '2', ..], ['3', ..], ...](即field中每个值为一个list，list中的值会被依次转换。)

        :param inplace: 如果为True，则将新生成值替换本field。否则返回list。
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
        将本field中的值调用float(cell). 支持field中内容为以下两种情况(1)['1', '2', ...](即field中每个值为str的)，
            (2) [['1', '2', ..], ['3', ..], ...](即field中每个值为一个list，list中的值会被依次转换。)

        :param inplace: 如果为True，则将新生成值替换本field。否则返回list。
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
        将本field中的值调用bool(cell). 支持field中内容为以下两种情况(1)['1', '2', ...](即field中每个值为str的)，
            (2) [['1', '2', ..], ['3', ..], ...](即field中每个值为一个list，list中的值会被依次转换。)

        :param inplace: 如果为True，则将新生成值替换本field。否则返回list。
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
        将本field中的值调用cell.lower(). 支持field中内容为以下两种情况(1)['1', '2', ...](即field中每个值为str的)，
            (2) [['1', '2', ..], ['3', ..], ...](即field中每个值为一个list，list中的值会被依次转换。)

        :param inplace: 如果为True，则将新生成值替换本field。否则返回list。
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
        将本field中的值调用cell.lower(). 支持field中内容为以下两种情况(1)['1', '2', ...](即field中每个值为str的)，
            (2) [['1', '2', ..], ['3', ..], ...](即field中每个值为一个list，list中的值会被依次转换。)

        :param inplace: 如果为True，则将新生成值替换本field。否则返回list。
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

    def value_count(self):
        r"""
        返回该field下不同value的数量。多用于统计label数量

        :return: Counter, key是label，value是出现次数
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
        当调用处理函数之后，决定是否要替换field。

        :param new_contents:
        :param inplace:
        :return: self或者生成的content
        """
        if inplace:
            self.content = new_contents
            return self
        else:
            return new_contents

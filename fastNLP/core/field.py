r"""
.. todo::
    doc
"""

__all__ = [
    "Padder",
    "AutoPadder",
    "EngChar2DPadder",
]

from abc import abstractmethod
from collections import Counter
from copy import deepcopy
from numbers import Number
from typing import Any

import numpy as np
import torch

from ._logger import logger
from .utils import _is_iterable


class SetInputOrTargetException(Exception):
    def __init__(self, msg, index=None, field_name=None):
        super().__init__(msg)
        self.msg = msg
        self.index = index  # 标示在哪个数据遭遇到问题了
        self.field_name = field_name  # 标示当前field的名称


class AppendToTargetOrInputException(Exception):
    def __init__(self, msg, index=None, field_name=None):
        super().__init__(msg)
        self.msg = msg
        self.index = index  # 标示在哪个数据遭遇到问题了
        self.field_name = field_name  # 标示当前field的名称


class FieldArray:
    def __init__(self, name, content, is_target=False, is_input=False, padder=None, ignore_type=False,
                 use_1st_ins_infer_dim_type=True):
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
        self._ignore_type = ignore_type
        #  根据input的情况设置input，target等
        self._cell_ndim = None  # 多少维度， 如果value是1, dim为0; 如果value是[1, 2], dim=2
        self.dtype = None  # 最内层的element都是什么类型的
        self._use_1st_ins_infer_dim_type = bool(use_1st_ins_infer_dim_type)
        self._is_input = False
        self._is_target = False
        
        if is_input:
            self.is_input = is_input
        if is_target:
            self.is_target = is_target
        
        if padder is None:
            padder = AutoPadder(pad_val=0)
        else:
            assert isinstance(padder, Padder), "padder must be of type fastNLP.Padder."
            padder = deepcopy(padder)
        self.set_padder(padder)
    
    @property
    def ignore_type(self):
        return self._ignore_type
    
    @ignore_type.setter
    def ignore_type(self, value):
        if value:
            self._cell_ndim = None
            self.dtype = None
        self._ignore_type = value
    
    @property
    def is_input(self):
        return self._is_input
    
    @is_input.setter
    def is_input(self, value):
        r"""
            当 field_array.is_input = True / False 时被调用
        """
        # 如果(value为True)且(_is_input和_is_target都是False)且(ignore_type为False)
        if value is True and \
                self._is_target is False and \
                self._ignore_type is False:
            self._check_dtype_and_ndim(only_check_1st_ins_dim_type=self._use_1st_ins_infer_dim_type)
        if value is False and self._is_target is False:
            self.dtype = None
            self._cell_ndim = None
        self._is_input = value
    
    @property
    def is_target(self):
        return self._is_target
    
    @is_target.setter
    def is_target(self, value):
        r"""
        当 field_array.is_target = True / False 时被调用
        """
        if value is True and \
                self._is_input is False and \
                self._ignore_type is False:
            self._check_dtype_and_ndim(only_check_1st_ins_dim_type=self._use_1st_ins_infer_dim_type)
        if value is False and self._is_input is False:
            self.dtype = None
            self._cell_ndim = None
        self._is_target = value
    
    def _check_dtype_and_ndim(self, only_check_1st_ins_dim_type=True):
        r"""
        检查当前content所有的element是否是同一个类型，且是否每个元素具有相同的维度。通过的话，设置_cell_ndim与_ele_type属性；没有
            通过将直接报错.

        :param bool only_check_1st_ins_dim_type: 是否只检查第一个元素的type和dim
        :return:
        """
        cell_0 = self.content[0]
        index = 0
        try:
            type_0, dim_0 = _get_ele_type_and_dim(cell_0)
            if not only_check_1st_ins_dim_type:
                for cell in self.content[1:]:
                    index += 1
                    type_i, dim_i = _get_ele_type_and_dim(cell)
                    if type_i != type_0:
                        raise SetInputOrTargetException(
                            "Type:{} in index {} is different from the first element with type:{}."
                            ".".format(type_i, index, type_0))
                    if dim_0 != dim_i:
                        raise SetInputOrTargetException(
                            "Dimension:{} in index {} is different from the first element with "
                            "dimension:{}.".format(dim_i, index, dim_0))
            self._cell_ndim = dim_0
            self.dtype = type_0
        except SetInputOrTargetException as e:
            e.index = index
            raise e
    
    def append(self, val: Any):
        r"""
        :param val: 把该val append到fieldarray。
        :return:
        """
        if (self._is_target or self._is_input) and self._ignore_type is False and not self._use_1st_ins_infer_dim_type:
            type_, dim_ = _get_ele_type_and_dim(val)
            if self.dtype != type_:
                raise AppendToTargetOrInputException(f"Value(type:{type_}) are of different types with "
                                                     f"previous values(type:{self.dtype}).")
            if self._cell_ndim != dim_:
                raise AppendToTargetOrInputException(f"Value(dim:{dim_}) are of different dimensions with "
                                                     f"previous values(dim:{self._cell_ndim}).")
            self.content.append(val)
        else:
            self.content.append(val)
    
    def pop(self, index):
        r"""
        删除该field中index处的元素
        :param int index: 从0开始的数据下标。
        :return:
        """
        self.content.pop(index)
    
    def __getitem__(self, indices):
        return self.get(indices, pad=False)
    
    def __setitem__(self, idx, val):
        assert isinstance(idx, int)
        if (self._is_target or self._is_input) and self.ignore_type is False:  # 需要检测类型
            type_, dim_ = _get_ele_type_and_dim(val)
            if self.dtype != type_:
                raise RuntimeError(f"Value(type:{type_}) are of different types with "
                                   f"other values(type:{self.dtype}).")
            if self._cell_ndim != dim_:
                raise RuntimeError(f"Value(dim:{dim_}) are of different dimensions with "
                                   f"previous values(dim:{self._cell_ndim}).")
        self.content[idx] = val
    
    def get(self, indices, pad=True):
        r"""
        根据给定的indices返回内容。

        :param int,List[int] indices: 获取indices对应的内容。
        :param bool pad: 是否对返回的结果进行padding。仅对: (1) indices为List[int]; (2)padder不为None; (3)field设置了input
            或target，有效
        :return: 根据给定的indices返回的内容，可能是单个值或ndarray
        """
        if isinstance(indices, int):
            return self.content[indices]

        contents = [self.content[i] for i in indices]
        if self.padder is None or pad is False:
            return np.array(contents)
        elif self.is_input or self.is_target:
            return self.pad(contents)
        else:
            return np.array(contents)
    
    def pad(self, contents):
        r"""
        传入list的contents，将contents使用padder进行padding，contents必须为从本FieldArray中取出的。

        :param list contents:
        :return:
        """
        return self.padder(contents, field_name=self.name, field_ele_dtype=self.dtype, dim=self._cell_ndim)
    
    def set_padder(self, padder):
        r"""
        设置padder，在这个field进行pad的时候用这个padder进行pad，如果为None则不进行pad。

        :param padder: :class:`~fastNLP.Padder` 类型，设置为None即删除padder。
        """
        if padder is not None:
            assert isinstance(padder, Padder), "padder must be of type Padder."
            self.padder = deepcopy(padder)
        else:
            self.padder = None
    
    def set_pad_val(self, pad_val):
        r"""
        修改padder的pad_val.

        :param int pad_val: 该field的pad值设置为该值。
        """
        if self.padder is not None:
            self.padder.set_pad_val(pad_val)
        return self
    
    def __len__(self):
        r"""
        Returns the size of FieldArray.

        :return int length:
        """
        return len(self.content)
    
    def to(self, other):
        r"""
        将other的属性复制给本FieldArray(other必须为FieldArray类型).
        属性包括 is_input, is_target, padder, ignore_type

        :param  other: :class:`~fastNLP.FieldArray` 从哪个field拷贝属性
        :return: :class:`~fastNLP.FieldArray`
        """
        assert isinstance(other, FieldArray), "Only supports fastNLP.FieldArray type, not {}.".format(type(other))
        
        self.ignore_type = other.ignore_type
        self.is_input = other.is_input
        self.is_target = other.is_target
        self.padder = other.padder
        
        return self
    
    def split(self, sep: str = None, inplace: bool = True):
        r"""
        依次对自身的元素使用.split()方法，应该只有当本field的元素为str时，该方法才有用。将返回值

        :param sep: 分割符，如果为None则直接调用str.split()。
        :param inplace: 如果为True，则将新生成值替换本field。否则返回list。
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
                logger.error(f"Exception happens when process value in index {index}.")
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
                logger.error(f"Exception happens when process value in index {index}.")
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
                logger.error(f"Exception happens when process value in index {index}.")
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
                logger.error(f"Exception happens when process value in index {index}.")
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
                logger.error(f"Exception happens when process value in index {index}.")
                raise e
        return self._after_process(new_contents, inplace=inplace)
    
    def value_count(self):
        r"""
        返回该field下不同value的数量。多用于统计label数量

        :return: Counter, key是label，value是出现次数
        """
        count = Counter()
        
        def cum(cell):
            if _is_iterable(cell) and not isinstance(cell, str):
                for cell_ in cell:
                    cum(cell_)
            else:
                count[cell] += 1
        
        for cell in self.content:
            cum(cell)
        return count
    
    def _after_process(self, new_contents, inplace):
        r"""
        当调用处理函数之后，决定是否要替换field。

        :param new_contents:
        :param inplace:
        :return: self或者生成的content
        """
        if inplace:
            self.content = new_contents
            try:
                self.is_input = self.is_input
                self.is_target = self.is_input
            except SetInputOrTargetException as e:
                logger.error("The newly generated field cannot be set as input or target.")
                raise e
            return self
        else:
            return new_contents


def _get_ele_type_and_dim(cell: Any, dim=0):
    r"""
    识别cell的类别与dimension的数量

    numpy scalar type:https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.scalars.html
    :param cell:
    :param dim:
    :return:
    """
    if isinstance(cell, (str, Number, np.bool_)):
        if hasattr(cell, 'dtype'):
            return cell.dtype.type, dim
        return type(cell), dim
    elif isinstance(cell, list):
        dim += 1
        res = [_get_ele_type_and_dim(cell_i, dim) for cell_i in cell]
        types = set([i for i, j in res])
        dims = set([j for i, j in res])
        if len(types) > 1:
            raise SetInputOrTargetException("Mixed types detected: {}.".format(list(types)))
        elif len(types) == 0:
            raise SetInputOrTargetException("Empty value encountered.")
        if len(dims) > 1:
            raise SetInputOrTargetException("Mixed dimension detected: {}.".format(list(dims)))
        return types.pop(), dims.pop()
    elif isinstance(cell, torch.Tensor):
        return cell.dtype, cell.dim() + dim  # 如果是torch.mean的结果是0
    elif isinstance(cell, np.ndarray):
        if cell.dtype != np.dtype('O'):  # 如果不是object的话说明是well-formatted的了
            return cell.dtype.type, cell.ndim + dim  # dtype.type返回的会是np.int32, np.float等
        # 否则需要继续往下iterate
        dim += 1
        res = [_get_ele_type_and_dim(cell_i, dim) for cell_i in cell]
        types = set([i for i, j in res])
        dims = set([j for i, j in res])
        if len(types) > 1:
            raise SetInputOrTargetException("Mixed types detected: {}.".format(list(types)))
        elif len(types) == 0:
            raise SetInputOrTargetException("Empty value encountered.")
        if len(dims) > 1:
            raise SetInputOrTargetException("Mixed dimension detected: {}.".format(list(dims)))
        return types.pop(), dims.pop()
    else:  # 包含tuple, set, dict以及其它的类型
        raise SetInputOrTargetException(f"Cannot process type:{type(cell)}.")


class Padder:
    r"""
    所有padder都需要继承这个类，并覆盖__call__方法。
    用于对batch进行padding操作。传入的element是inplace的，即直接修改element可能导致数据变化，建议inplace修改之前deepcopy一份。

    .. py:function:: __call__(self, contents, field_name, field_ele_dtype):
    
    """
    
    def __init__(self, pad_val=0, **kwargs):
        r"""
        
        :param List[Any] contents: 传入的element是inplace的，即直接修改element可能导致数据变化，建议inplace修改之前
            deepcopy一份。
        :param str, field_name: field的名称。
        :param np.int64,np.float64,np.str,None, field_ele_dtype: 该field的内层元素的类型。如果该field的ignore_type为True，该这个值为None。
        :return: np.array([padded_element])
        """
        self.pad_val = pad_val
    
    def set_pad_val(self, pad_val):
        self.pad_val = pad_val

    def get_pad_val(self):
        return self.pad_val

    @abstractmethod
    def __call__(self, contents, field_name, field_ele_dtype, dim: int):
        r"""
        传入的是List内容。假设有以下的DataSet。

        :param List[Any] contents: 传入的element是inplace的，即直接修改element可能导致数据变化，建议inplace修改之前
            deepcopy一份。
        :param str, field_name: field的名称。
        :param np.int64,np.float64,np.str,None, field_ele_dtype: 该field的内层元素的类型。如果该field的ignore_type为True，
            该这个值为None。
        :param dim: 这个field的维度。当ignore_type为True时，该值为None
        :return: np.array([padded_element])

        Example::

            from fastNLP import DataSet
            from fastNLP import Instance
            dataset = DataSet()
            dataset.append(Instance(sent='this is a demo', length=4,
                                    chars=[['t', 'h', 'i', 's'], ['i', 's'], ['a'], ['d', 'e', 'm', 'o']]))
            dataset.append(Instance(sent='another one', length=2,
                                    chars=[['a', 'n', 'o', 't', 'h', 'e', 'r'], ['o', 'n', 'e']]))
            如果调用
            batch = dataset.get([0,1], pad=True)
            sent这个field的padder的__call__会接收到的内容会是
                [
                    'this is a demo',
                    'another one'
                ]

            length这个field的padder的__call__会接收到的内容会是
                [4, 2]

            chars这个field的padder的__call__会接收到的内容会是
                [
                    [['t', 'h', 'i', 's'], ['i', 's'], ['a'], ['d', 'e', 'm', 'o']],
                    [['a', 'n', 'o', 't', 'h', 'e', 'r'], ['o', 'n', 'e']]
                ]

        即把每个instance中某个field的内容合成一个List传入

        """
        raise NotImplementedError


class AutoPadder(Padder):
    r"""
    根据contents的数据自动判定是否需要做padding。

    1 如果元素类型(元素类型是指field中最里层元素的数据类型, 可以通过FieldArray.dtype查看，比如['This', 'is', ...]的元素类
    型为str, [[1,2], ...]的元素类型为int)的数据不为数值类型则不会进行pad

    2 如果元素类型为数值类型,比如np.int64, np.float64, int, float, torch.int64等

        2.1 如果该field的内容为数值类型(包括int, float等)，比如为seq_len, 则不进行padding

        2.2 如果该field的内容等价于一维list, 那么会将Batch中的List pad为一样长。

        2.3 如果该field的内容等价于二维list，那么会按照英语character padding的方式进行padding。如果是character padding建议使用
            :class: fastNLP.EngChar2DPadder.

        2.4 如果该field的内容等价于三维list，则如果每个instance在每个维度上相等，会组成一个batch的tensor返回，这种情况应该是为图片
            的情况。

    3 其它情况不进行处理，返回一个np.array类型。
    """
    
    def __init__(self, pad_val=0):
        super().__init__(pad_val=pad_val)
    
    def __call__(self, contents, field_name, field_ele_dtype, dim):
        if field_ele_dtype:
            if dim > 3:
                return np.array(contents)
            if isinstance(field_ele_dtype, type) and \
                    (issubclass(field_ele_dtype, np.number) or issubclass(field_ele_dtype, Number)):
                if dim == 0:
                    array = np.array(contents, dtype=field_ele_dtype)
                elif dim == 1:
                    max_len = max(map(len, contents))
                    array = np.full((len(contents), max_len), self.pad_val, dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        array[i, :len(content_i)] = content_i
                elif dim == 2:
                    max_len = max(map(len, contents))
                    max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                                        content_i in contents])
                    array = np.full((len(contents), max_len, max_word_len), self.pad_val, dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        for j, content_ii in enumerate(content_i):
                            array[i, j, :len(content_ii)] = content_ii
                else:
                    shape = np.shape(contents)
                    if len(shape) == 4:  # 说明各dimension是相同的大小
                        array = np.array(contents, dtype=field_ele_dtype)
                    else:
                        raise RuntimeError(
                            f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                return array
            elif str(field_ele_dtype).startswith('torch'):
                if dim == 0:
                    tensor = torch.tensor(contents).to(field_ele_dtype)
                elif dim == 1:
                    max_len = max(map(len, contents))
                    tensor = torch.full((len(contents), max_len), fill_value=self.pad_val, dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        tensor[i, :len(content_i)] = content_i.clone().detach()
                elif dim == 2:
                    max_len = max(map(len, contents))
                    max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                                        content_i in contents])
                    tensor = torch.full((len(contents), max_len, max_word_len), fill_value=self.pad_val,
                                        dtype=field_ele_dtype)
                    for i, content_i in enumerate(contents):
                        for j, content_ii in enumerate(content_i):
                            tensor[i, j, :len(content_ii)] = content_ii.clone().detach()
                else:
                    shapes = set([np.shape(content_i) for content_i in contents])
                    if len(shapes) > 1:
                        raise RuntimeError(
                            f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                    shape = shapes.pop()
                    if len(shape) == 3:
                        tensor = torch.full([len(contents)] + list(shape), fill_value=self.pad_val,
                                            dtype=field_ele_dtype)
                        for i, content_i in enumerate(contents):
                            tensor[i] = content_i.clone().detach().to(field_ele_dtype)
                    else:
                        raise RuntimeError(
                            f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                return tensor
            else:
                return np.array(contents)  # 不进行任何操作
        else:
            return np.array(contents)


class EngChar2DPadder(Padder):
    r"""
    用于为英语执行character级别的2D padding操作。对应的field内容应该类似[['T', 'h', 'i', 's'], ['a'], ['d', 'e', 'm', 'o']]，
    但这个Padder只能处理index为int的情况。

    padded过后的batch内容，形状为(batch_size, max_sentence_length, max_word_length). max_sentence_length为这个batch中最大句
    子长度；max_word_length为这个batch中最长的word的长度::

        from fastNLP import DataSet
        from fastNLP import EngChar2DPadder
        from fastNLP import Vocabulary
        dataset = DataSet({'sent': ['This is the first demo', 'This is the second demo']})
        dataset.apply(lambda ins:[list(word) for word in ins['sent'].split()], new_field_name='chars')
        vocab = Vocabulary()
        vocab.from_dataset(dataset, field_name='chars')
        vocab.index_dataset(dataset, field_name='chars')
        dataset.set_input('chars')
        padder = EngChar2DPadder()
        dataset.set_padder('chars', padder)  # chars这个field的设置为了EnChar2DPadder

    """
    
    def __init__(self, pad_val=0, pad_length=0):
        r"""
        :param pad_val: int, pad的位置使用该index
        :param pad_length: int, 如果为0则取一个batch中最大的单词长度作为padding长度。如果为大于0的数，则将所有单词的长度
            都pad或截取到该长度.
        """
        super().__init__(pad_val=pad_val)
        
        self.pad_length = pad_length
    
    def __call__(self, contents, field_name, field_ele_dtype, dim):
        r"""
        期望输入类似于
        [
            [[0, 2], [2, 3, 4], ..],
            [[9, 8, 2, 4], [1, 2,], ...],
            ....
        ]

        :param contents:
        :param field_name:
        :param field_ele_dtype
        :return:
        """
        if field_ele_dtype not in (np.int64, np.float64, int, float):
            raise TypeError('dtype of Field:{} should be np.int64 or np.float64 to do 2D padding, get {}.'.format(
                field_name, field_ele_dtype
            ))
        assert dim == 2, f"Field:{field_name} has {dim}, EngChar2DPadder only supports input with 2 dimensions."
        if self.pad_length < 1:
            max_char_length = max([max(len(char_lst) for char_lst in word_lst) for word_lst in contents])
        else:
            max_char_length = self.pad_length
        max_sent_length = max(len(word_lst) for word_lst in contents)
        batch_size = len(contents)
        dtype = type(contents[0][0][0])
        
        padded_array = np.full((batch_size, max_sent_length, max_char_length), fill_value=self.pad_val,
                               dtype=dtype)
        for b_idx, word_lst in enumerate(contents):
            for c_idx, char_lst in enumerate(word_lst):
                chars = char_lst[:max_char_length]
                padded_array[b_idx, c_idx, :len(chars)] = chars
        
        return padded_array

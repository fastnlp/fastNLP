__all__ = [
    'AutoCollator',
    'Collator',
]


from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Callable, Union
from numbers import Number
import warnings

import numpy as np

from fastNLP.envs.imports import _NEED_IMPORT_PADDLE, _NEED_IMPORT_TORCH

if _NEED_IMPORT_PADDLE:
    import paddle

if _NEED_IMPORT_TORCH:
    import torch


class ApplyResultException(Exception):
    def __init__(self, msg, index=None):
        super().__init__(msg)
        self.msg = msg
        self.index = index  # 标示在哪个数据遭遇到问题了


class SetInputOrTargetException(Exception):
    def __init__(self, msg, index=None, field_name=None):
        super().__init__(msg)
        self.msg = msg
        self.index = index  # 标示在哪个数据遭遇到问题了
        self.field_name = field_name  # 标示当前 field 的名称


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
        return cell.dtype, cell.dim() + dim  # 如果是 torch.mean 的结果是0

    elif isinstance(cell, paddle.Tensor):
        return cell.dtype, cell.dim() + dim

    elif isinstance(cell, np.ndarray):
        if cell.dtype != np.dtype('O'):  # 如果不是 object 的话说明是 well-formatted 的了
            return cell.dtype.type, cell.ndim + dim  # dtype.type 返回的会是 np.int32, np.float 等
        # 否则需要继续往下 iterate
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

    else:  # 包含 tuple, set, dict 以及其它的类型
        raise SetInputOrTargetException(f"Cannot process type:{type(cell)}.")


def _get_ds_type_dim(ds: dict):
    # 获取数据集第一行的 field 内部函数的类型和维度
    field_dtype, field_dim = {}, {}
    for field_name, field_content in ds.items():
        type_0, dim_0 = _get_ele_type_and_dim(field_content)
        field_dtype[field_name], field_dim[field_name] = type_0, dim_0
    return field_dtype, field_dim


class Collator(metaclass=ABCMeta):
    r"""
        辅助DataLoader管理collate_fn的类

    """

    def __init__(self):
        super(Collator, self).__init__()
        self.collate_fn = []

    @abstractmethod
    def __call__(self, ins_lst: List) -> Any:
        raise NotImplementedError

    @abstractmethod
    def set_pad_val(self, *field_names: str, value=0):
        raise NotImplementedError


class _MultiCollator:
    """
    管理所有collator的容器，
    遵循覆盖原则，后加入的collate_fn会覆盖之前处理的数据。
    """

    def __init__(self, collate_fns: Union[Callable, List[Callable], None]):

        if collate_fns is None:
            collate_fns = []

        if isinstance(collate_fns, Callable):
            collate_fns = [collate_fns]

        self._collators: list = collate_fns

    def __call__(self, ins_lst) -> Dict:
        out, list_out = {}, []
        for idx, _collate_fn in enumerate(self._collators):
            res = _collate_fn(ins_lst)
            if isinstance(res, Dict):
                out.update(res)
            else:
                list_out.append(res)
            # else:
            #     raise ValueError(f"the return type of collate_fn {idx} is {type(res)}, but require is dict")
        if len(out) > 0 and len(list_out) > 0:
            raise ValueError("the return of collate_fns is not the same, must be dict or list")
        if len(list_out) == 1:
            list_out = list_out[-1]
        # print(list_out)
        return out if len(out) > 0 else list_out

    def get_collators(self):
        return self._collators

    def add_collator(self, collator: Callable):
        self._collators.append(collator)

    def set_as_numpy(self, as_numpy: bool):
        """
        存在AutoCollator时，as_numpy控制其返回值的类型

        :param as_numpy:
        :return:
        """
        for collator in self._collators:
            if isinstance(collator, AutoCollator):
                collator.set_as_numpy(as_numpy)
        return self

    def set_pad_val(self, *field_names, val=0):
        """
        存在AutoCollator时，设置field_name的padding值

        :param field_names: 数据集的field名
        :param val: padding的值
        :return:
        """
        flag = True
        for collator in self._collators:
            if isinstance(collator, AutoCollator):
                collator.set_pad_val(*field_names, val=val)
                flag = False
        if flag:
            warnings.warn("AutoCollator is remove, set_padding is unavailable!!")
        return self

    def set_input(self, *field_names):
        """
        设置AutoCollator需要的field_names,未被设置默认过滤掉

        :param field_names:
        :return:
        """
        flag = True
        for collator in self._collators:
            if isinstance(collator, AutoCollator):
                collator.set_input(*field_names)
                flag = False
        if flag:
            warnings.warn("AutoCollator is removed, set_input is unavailable!!")
        return self


class AutoCollator(Collator):

    def __init__(self, as_numpy: bool):
        super(AutoCollator, self).__init__()
        self.pad_field_value = {}  # field padding 自定义的 padding 值, 默认为0
        self.need_inputs = []  # 需要的 field name
        self.field_dtypes = None  # 每列数据单元的 dtype 类型
        self.field_dims = None  # 每列数据单元维度
        self.as_numpy = as_numpy

    def __call__(self, ins_lst: List[Dict]) -> dict:
        if len(self.need_inputs) == 0:
            raise ValueError({"set_inputs is None, you should use set_inputs method first!!"})
        # 第一种情况，设置了 set_input 的值
        # 第二种情况， 根据数据的类型的判断是否 padding
        if self.field_dtypes is None and self.field_dims is None:
            self.field_dtypes, self.field_dims = _get_ds_type_dim(ins_lst[0])

        pack_ins_lst, pad_ins_lst = {field_name: []
                                     for field_name in ins_lst[0].keys() if field_name in self.need_inputs}, {}
        # 将 list 列表内数据按列名打包
        for per_ins in ins_lst:
            for field_name, _field_content in per_ins.items():
                if field_name in self.need_inputs:
                    pack_ins_lst[field_name].append(_field_content)

        pad_field_kv = {field_name: 0 for field_name in self.need_inputs}
        pad_field_kv.update(self.pad_field_value)
        self.pad_field_value = pad_field_kv

        if len(self.pad_field_value.keys()) > 0:
            # 去掉不需要 pad 的列，如果 set_input 的列不存在则忽略
            drop_field_names = []
            for k, v in self.pad_field_value.items():
                if v is None:
                    drop_field_names.append(k)

            # drop_field_names = list(set(list(ins_lst[0].keys())) - set(drop_fields))
            for field_name in drop_field_names:
                field_array = pack_ins_lst.pop(field_name)
                pad_ins_lst[field_name] = np.array(field_array)

            for field_name, field_array in pack_ins_lst.items():
                content = pad_content(field_array, field_name, self.field_dtypes[field_name],
                                      self.field_dims[field_name],
                                      self.pad_field_value[field_name],
                                      as_numpy=self.as_numpy)
                pad_ins_lst[field_name] = content

        # else:
        #     # 取出每列的数据，根据类型判断是否能 pad
        #     for field_name, field_array in pack_ins_lst.items():
        #         pad_field_array = pad_content(field_array, field_name, self.field_dtypes[field_name],
        #                                       self.field_dims[field_name],
        #                                       pad_val=0, as_numpy=self.as_numpy)
        #         pad_ins_lst[field_name] = pad_field_array

        return pad_ins_lst

    def set_pad_val(self, *field_names, val=0):
        for field_name in field_names:
            self.pad_field_value[field_name] = val

    def set_as_numpy(self, as_numpy: bool):
        self.as_numpy = as_numpy

    def set_input(self, *field_names):
        for field_name in field_names:
            self.need_inputs.append(field_name)


def pad_content(content, field_name: str, field_type, field_dim: int, pad_val: int, as_numpy: bool):

    if field_type:
        # 不处理， 返回 np.array 类型
        if field_dim > 3:
            return np.array(content)
        # 元素类型为数值类型 np.int64, np.float64, int, float 等
        if isinstance(field_type, type) and \
                (issubclass(field_type, np.number) or issubclass(field_type, Number)):
            if field_dim == 0:
                array = np.array(content, dtype=field_type)
            elif field_dim == 1:
                max_len = max(map(len, content))
                array = np.full((len(content), max_len), pad_val, dtype=field_type)
                for i, content_i in enumerate(content):
                    array[i, :len(content_i)] = content_i
            elif field_dim == 2:
                max_len = max(map(len, content))
                max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                                    content_i in content])
                array = np.full((len(content), max_len, max_word_len), pad_val, dtype=field_type)
                for i, content_i in enumerate(content):
                    for j, content_ii in enumerate(content_i):
                        array[i, j, :len(content_ii)] = content_ii
            else:
                shape = np.shape(content)
                if len(shape) == 4:  # 说明各 dimension 是相同的大小
                    array = np.array(content, dtype=field_type)
                else:
                    raise RuntimeError(
                        f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
            if as_numpy is False:
                array = torch.tensor(array)
            return array
        # 元素类型为数值类型 torch.float 等
        elif str(field_type).startswith('torch'):
            if field_dim == 0:
                tensor = torch.tensor(content).to(field_type)
            elif field_dim == 1:
                max_len = max(map(len, content))
                tensor = torch.full((len(content), max_len), fill_value=pad_val, dtype=field_type)
                for i, content_i in enumerate(content):
                    tensor[i, :len(content_i)] = content_i.clone().detach()
            elif field_dim == 2:
                max_len = max(map(len, content))
                max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                                    content_i in content])
                tensor = torch.full((len(content), max_len, max_word_len), fill_value=pad_val,
                                    dtype=field_type)
                for i, content_i in enumerate(content):
                    for j, content_ii in enumerate(content_i):
                        tensor[i, j, :len(content_ii)] = content_ii.clone().detach()
            else:
                shapes = set([np.shape(content_i) for content_i in content])
                if len(shapes) > 1:
                    raise RuntimeError(
                        f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                shape = shapes.pop()
                if len(shape) == 3:
                    tensor = torch.full([len(content)] + list(shape), fill_value=pad_val,
                                        dtype=field_type)
                    for i, content_i in enumerate(content):
                        tensor[i] = content_i.clone().detach().to(field_type)
                else:
                    raise RuntimeError(
                        f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
            return tensor
        # TODO 增加jittor/paddle？
        elif str(field_type).startswith('paddle'):
            if field_dim == 0:
                tensor = paddle.Tensor(content).to(field_type)
            elif field_dim == 1:
                max_len = max(map(len, content))
                tensor = paddle.full((len(content), max_len), fill_value=pad_val, dtype=field_type)
                for i, content_i in enumerate(content):
                    tensor[i, :len(content_i)] = content_i.clone().detach()
            elif field_dim == 2:
                max_len = max(map(len, content))
                max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
                                    content_i in content])
                tensor = paddle.full((len(content), max_len, max_word_len), fill_value=pad_val,
                                     dtype=field_type)
                for i, content_i in enumerate(content):
                    for j, content_ii in enumerate(content_i):
                        tensor[i, j, :len(content_ii)] = content_ii.clone().detach()
            else:
                shapes = set([np.shape(content_i) for content_i in content])
                if len(shapes) > 1:
                    raise RuntimeError(
                        f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
                shape = shapes.pop()
                if len(shape) == 3:
                    tensor = paddle.full([len(content)] + list(shape), fill_value=pad_val,
                                         dtype=field_type)
                    for i, content_i in enumerate(content):
                        tensor[i] = content_i.clone().detach().to(field_type)
                else:
                    raise RuntimeError(
                        f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
            return tensor

        else:
            return np.array(content)  # 不进行任何操作
    else:
        return np.array(content)

__all__ = [
    'Collator'
]

from typing import List, Union, Dict, Callable, Sequence, Mapping
import os
import sys
import inspect

from fastNLP.core.log import logger
from .padders.get_padder import get_padder

import re

from .utils import unpack_batch_mapping, unpack_batch_nested_mapping, pack_batch_nested_mapping, unpack_batch_sequence, \
    pack_batch_sequence

sequence_idx_str = re.compile(r'^_\d+$')  # 形如_0, _1
SUPPORTED_BACKENDS = ['torch', 'jittor', 'paddle', 'numpy', 'raw', 'auto', None]
CHECK_BACKEND = ['torch', 'jittor', 'paddle']  # backend 为 auto 时 检查是否是这些 backend


def _get_backend() -> str:
    """
    当 Collator 的 backend 为 None 的时候如何，通过这个函数自动判定其 backend 。判断方法主要为以下两个：
    （1）尝试通过向上寻找当前 collator 的 callee 对象，根据 callee 对象寻找。然后使用 '/site-packages/{backend}' 来寻找是否是
        某个 backend 的 dataloader 。
    （2）如果方式（1）没找，则通过分析 sys.modules 中的内容进行寻找。

    如果都没有找到则返回 numpy 。
    :return:
    """
    def _check_module(module):
        """
        检查该 module 是否含有 某个 backend 的特征

        :param module: module 对象
        :return:
        """
        catch_backend = []
        try:
            file = module.__file__
            for backend in CHECK_BACKEND:
                if f'{os.sep}site-packages{os.sep}{backend}' in file:
                    catch_backend = [backend, file]
        except:
            pass
        return catch_backend

    currentframe = inspect.currentframe()
    # 方式（1）
    catch_backend = []
    for i in range(100):
        currentframe = currentframe.f_back
        if currentframe is not None:
            module = inspect.getmodule(currentframe)
            if module is not None:
                catch_backend = _check_module(module)
                if len(catch_backend):  # 主要捕获到一个就结束吧
                    break
        else:
            break
    if len(catch_backend):
        logger.debug(f"Find a file named:{catch_backend[1]} from stack contains backend:{catch_backend[0]}.")
        return catch_backend[0]

    # 方式 (2)
    for backend in CHECK_BACKEND:
        if backend in sys.modules:
            logger.debug(f"sys.modules contains backend:{backend}.")
            return backend
    for key, module in sys.modules.items():
        catch_backend = _check_module(module)
        if catch_backend:
            break
    if len(catch_backend):
        logger.debug(f"Find a module file named:{catch_backend[1]} from sys.modules contains backend:{catch_backend[0]}.")
        return catch_backend[0]

    return 'numpy'


class Collator:
    def __init__(self, backend='auto'):
        """
        用于 pad 数据的对象。会自动将所有能够 pad （由 fastNLP 根据数据判定能否 pad ）的数据都进行 pad 操作，默认 pad 的值为 0。
            可使用 set_pad() 函数调整。如果有些 field 不想输出，可以使用 set_ignore() 函数进行设置。Collator 在第一次进行 pad 的
            时候自动根据设置以及数据情况，为每个 field 获取一个 padder ，在之后的每次调用中，都将使用对应的 Padder 给对应的 field 。

        :param backend: 对于可以 pad 的 field，使用哪种 tensor，支持 ['torch','jittor','paddle','numpy','raw', auto, None]。
            若为 'auto' ，则在进行 pad 的时候会根据调用的环境决定其 backend 。该参数对不能进行 pad 的数据没用影响，不能 pad
            的数据返回一定是 list 。
        """
        self.unpack_batch_func = None
        self.pack_batch_func = None
        self.ignore_fields = set()
        self.padders = {}
        self.input_fields = {}
        self.batch_data_type = None  # 只能是 d ，s ，l 三种，分别对应输入的batch的每个sample为 dict, single，list。
        self.set_backend(backend)

    def __call__(self, batch)->Union[List, Dict]:
        """
        batch可能存在三种可能性
            List[Dict], List[List], List[Sample]

        第一步：使用 unpack_batch_func 将相同 field 的内容打包到一个 list 中。
        第二步：使用每个 field 各自的 padder 进行 pad 。
        第三步：根据 batch 中每个 sample 的类型，返回也保证为该类型。

        第一次调用会根据当前 batch 数据决定使用哪个 unpack_batch_func ，这个函数的作用是把不同 sample 的同一个 field 的放入到一个
            list 中；同时也会决定 pack_batch_func，这个函数的作用是在返回 pad 好的 batch 之前，将 batch 恢复为 输入时一个 sample
            的类别。
        第一次调用会根据当前 field 决定对应的 Padder 。

        """
        if self.unpack_batch_func is None:
            # 决定使用哪个unpack_batch_func，让它都 return 回 dict 类型
            if self.batch_data_type is None:
                if isinstance(batch[0], Mapping):
                    self.batch_data_type = 'd'
                elif isinstance(batch[0], Sequence):  # 这里存在误判的风险
                    self.batch_data_type = 'l'
                else:
                    self.batch_data_type = 's'
                logger.debug(f"Since batch[0] has type:{type(batch[0])}, so the batch_data_type "
                             f"is `{self.batch_data_type}`.")
            if self.batch_data_type == 's':
                self.unpack_batch_func = lambda batch, ignore_fields: {'_single': batch}  # 不需要做任何调整
                self.pack_batch_func = lambda x: x['_single']
            elif self.batch_data_type == 'l':
                self.unpack_batch_func = unpack_batch_sequence
                self.pack_batch_func = pack_batch_sequence
            elif self.batch_data_type == 'd':
                if any([isinstance(v, Mapping) for v in batch[0].values()]):  # 可能存在 nested 的dict。{'a': {'b': xx}}->{('a', 'b'): value}
                    self.unpack_batch_func = unpack_batch_nested_mapping
                    self.pack_batch_func = pack_batch_nested_mapping
                else:
                    self.unpack_batch_func = unpack_batch_mapping
                    self.pack_batch_func = lambda x:x

        if self.unpack_batch_func is unpack_batch_nested_mapping:  # 比较特殊，需要防止继续往下延伸
            unpack_batch: Dict = self.unpack_batch_func(batch, self.ignore_fields, set(self.input_fields.keys()))
        else:
            unpack_batch:Dict = self.unpack_batch_func(batch, self.ignore_fields)  # 将各自 field 组成 batch 形式。

        pad_batch = {}
        if len(self.padders)==0:  # 第一次运行，准备 padder
            if self.backend == 'auto':  # 如果 backend 为 auto ，则尝试通过调用栈等自动获取 backend 。
                self.backend = _get_backend()

            for key in unpack_batch.keys():
                if key not in self.input_fields and key not in self.ignore_fields:
                    self.input_fields[key] = {'pad_val': 0, 'dtype': None, 'backend': self.backend}
                elif key in self.input_fields and self.input_fields[key]['backend'] == 'auto':
                    self.input_fields[key]['backend'] = self.backend

            for field_name, setting in self.input_fields.items():
                pad_fn = setting.get('pad_fn', None)
                if callable(pad_fn):
                    padder = pad_fn
                else:
                    backend = self.backend if setting['backend'] == 'auto' else setting['backend']
                    batch_field = unpack_batch.get(field_name)
                    padder = get_padder(batch_field=batch_field, pad_val=setting['pad_val'],
                                        dtype=setting['dtype'], backend=backend,
                                        field_name=field_name)
                self.padders[field_name] = padder
            if self.batch_data_type == 'l':
                self.padders = dict(sorted(self.padders.items(), key=lambda x:int(x[0][1:])))  # sort, 这样 _0, _1 能够保持顺序

        for key, padder in self.padders.items():
            batch = unpack_batch.get(key)
            pad_batch[key] = padder(batch)

        return self.pack_batch_func(pad_batch)  # 根据情况恢复成与输入一致的类型

    def set_pad(self, field_name:Union[str, tuple], pad_val:Union[int, float, None]=0, dtype=None, backend='auto',
                pad_fn:Callable=None) -> "Collator":
        """
        如果需要对某个 field 的内容进行特殊的调整，请使用这个函数。

        :param field_name: 需要调整的 field 的名称。如果 Dataset 的 __getitem__ 方法返回的是 dict 类型的，则可以直接使用对应的
            field 的 key 来表示，如果是 nested 的 dict，可以使用元组表示多层次的 key，例如 {'a': {'b': 1}} 中的使用 ('a', 'b');
            如果 __getitem__ 返回的是 Sequence 类型的，则可以使用 '_0', '_1' 表示序列中第 0 或 1 个元素。如果该 field 在数据中没
            有找到，则报错；如果 __getitem__ 返回的是就是整体内容，请使用 "_single" 。
        :param pad_val: 这个 field 的默认 pad 值。如果设置为 None，则表示该 field 不需要 pad , fastNLP 默认只会对可以 pad 的
            field 进行 pad，所以如果对应 field 本身就不是可以 pad 的形式，可以不需要主动设置为 None 。如果 backend 为 None ，该值
            无意义。
        :param dtype: 对于需要 pad 的 field ，该 field 的数据 dtype 应该是什么。
        :param backend: 可选['raw', 'numpy', 'torch', 'paddle', 'jittor', 'auto']，分别代表，输出为 list, numpy.ndarray,
            torch.Tensor, paddle.Tensor, jittor.Var 类型。若 pad_val 为 None ，该值无意义 。
        :param pad_fn: 指定当前 field 的 pad 函数，传入该函数则 pad_val, dtype, backend 等参数失效。pad_fn 的输入为当前 field 的
            batch 形式。 Collator 将自动 unbatch 数据，然后将各个 field 组成各自的 batch 。pad_func 的输入即为 field 的 batch
            形式，输出将被直接作为结果输出。
        :return: 返回 Collator 自身
        """
        self.padders.clear()  # 重新生成

        if self.batch_data_type is not None:
            if self.batch_data_type == 's':
                logger.debug("Set as single field mode.")
                self.input_fields.clear()
            elif self.batch_data_type == 'd':
                assert sequence_idx_str.match(field_name) is None, f"Field name:{field_name} will be recognized as list " \
                                                           f"index, but other field is set as dict mode."
            elif self.batch_data_type == 'l':
                assert sequence_idx_str.match(field_name) is not None, f"Other field is set as list mode. But the new " \
                                                                       f"field name is {field_name}."

        if field_name == '_single':
            self.batch_data_type = 's'
        elif isinstance(field_name, str) and sequence_idx_str.match(field_name):
            self.batch_data_type = 'l'
        else:
            self.batch_data_type = 'd'

        if field_name in self.ignore_fields:
            logger.warning(f"Field:{field_name} has been set as ignored before. It will not be ignored afterwards.")
        if backend is None:
            backend = self.backend
        else:
            assert backend in SUPPORTED_BACKENDS

        self.input_fields[field_name] = {'pad_val': pad_val, 'dtype': dtype, 'backend': backend, 'pad_fn': pad_fn}

        return self

    def set_backend(self, backend:str):
        """
        设置可以 pad 的 field 默认 pad 为什么类型的 tensor

        :param backend: 对于可以 pad 的 field，使用哪种 tensor，支持 ['torch','jittor','paddle','numpy','raw', 'auto', None]，
            若为 auto ，则在进行 pad 的时候会自动根据调用的环境决定其 backend 。
        :return:
        """
        assert backend in SUPPORTED_BACKENDS
        self.padders.clear()
        self.backend = backend

    def set_ignore(self, *field_names) -> "Collator":
        """
        如果有的内容不希望输出，可以在此处进行设置，被设置的 field 将在 batch 的输出中被忽略。
        Ex::
            collator.set_ignore('field1', 'field2')

        :param field_names: 需要忽略的 field 的名称。如果 Dataset 的 __getitem__ 方法返回的是 dict 类型的，则可以直接使用对应的
            field 的 key 来表示，如果是 nested 的 dict，可以使用元组来表示，例如 {'a': {'b': 1}} 中的使用 ('a', 'b'); 如果
            __getitem__ 返回的是 Sequence 类型的，则可以使用 '_0', '_1' 表示序列中第 0 或 1 个元素。
        :return: 返回 Collator 自身
        """
        for field_name in field_names:
            if field_name in self.input_fields:
                self.input_fields.pop(field_name)
                logger.warning(f"Field:{field_name} has been set as input before. It will be ignored afterwards.")
            self.padders.pop(field_name, None)  # 如果由的话，将它的 padder 扔掉。
            self.ignore_fields.add(field_name)

        return self








#
# from abc import ABCMeta, abstractmethod
# from typing import Any, Dict, List, Callable, Union, Tuple
# from numbers import Number
# import warnings
#
# import numpy as np
#
# from fastNLP.envs.imports import _NEED_IMPORT_PADDLE, _NEED_IMPORT_TORCH
#
# if _NEED_IMPORT_PADDLE:
#     import paddle
#
# if _NEED_IMPORT_TORCH:
#     import torch
#
#
# class ApplyResultException(Exception):
#     def __init__(self, msg, index=None):
#         super().__init__(msg)
#         self.msg = msg
#         self.index = index  # 标示在哪个数据遭遇到问题了
#
#
# class SetInputOrTargetException(Exception):
#     def __init__(self, msg, index=None, field_name=None):
#         super().__init__(msg)
#         self.msg = msg
#         self.index = index  # 标示在哪个数据遭遇到问题了
#         self.field_name = field_name  # 标示当前 field 的名称
#
#
# def _get_ele_type_and_dim(cell: Any, dim=0) -> Tuple[Any, int]:
#     r"""
#     识别cell的类别与dimension的数量
#
#     numpy scalar type:https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.scalars.html
#     :param cell:
#     :param dim:
#     :return:
#     """
#     if isinstance(cell, (str, Number, np.bool_)):
#         if hasattr(cell, 'dtype'):
#             return cell.dtype.type, dim
#         return type(cell), dim
#
#     elif isinstance(cell, list):
#         dim += 1
#         res = [_get_ele_type_and_dim(cell_i, dim) for cell_i in cell]
#         types = set([i for i, j in res])
#         dims = set([j for i, j in res])
#         if len(types) > 1:
#             raise SetInputOrTargetException("Mixed types detected: {}.".format(list(types)))
#         elif len(types) == 0:
#             raise SetInputOrTargetException("Empty value encountered.")
#         if len(dims) > 1:
#             raise SetInputOrTargetException("Mixed dimension detected: {}.".format(list(dims)))
#         return types.pop(), dims.pop()
#
#     elif isinstance(cell, torch.Tensor):
#         return cell.dtype, cell.dim() + dim  # 如果是 torch.mean 的结果是0
#
#     elif isinstance(cell, paddle.Tensor):
#         return cell.dtype, cell.dim() + dim
#
#     elif isinstance(cell, np.ndarray):
#         if cell.dtype != np.dtype('O'):  # 如果不是 object 的话说明是 well-formatted 的了
#             return cell.dtype.type, cell.ndim + dim  # dtype.type 返回的会是 np.int32, np.float 等
#         # 否则需要继续往下 iterate
#         dim += 1
#         res = [_get_ele_type_and_dim(cell_i, dim) for cell_i in cell]
#         types = set([i for i, j in res])
#         dims = set([j for i, j in res])
#         if len(types) > 1:
#             raise SetInputOrTargetException("Mixed types detected: {}.".format(list(types)))
#         elif len(types) == 0:
#             raise SetInputOrTargetException("Empty value encountered.")
#         if len(dims) > 1:
#             raise SetInputOrTargetException("Mixed dimension detected: {}.".format(list(dims)))
#         return types.pop(), dims.pop()
#
#     else:  # 包含 tuple, set, dict 以及其它的类型
#         raise SetInputOrTargetException(f"Cannot process type:{type(cell)}.")
#
#
# def _get_ds_type_dim(ds: dict):
#     # 获取数据集第一行的 field 内部函数的类型和维度
#     field_dtype, field_dim = {}, {}
#     for field_name, field_content in ds.items():
#         type_0, dim_0 = _get_ele_type_and_dim(field_content)
#         field_dtype[field_name], field_dim[field_name] = type_0, dim_0
#     return field_dtype, field_dim
#
#
# class Collator(metaclass=ABCMeta):
#     r"""
#         辅助DataLoader管理collate_fn的类
#
#     """
#
#     def __init__(self):
#         super(Collator, self).__init__()
#         self.collate_fn = []
#
#     @abstractmethod
#     def __call__(self, ins_lst: List) -> Any:
#         raise NotImplementedError
#
#     @abstractmethod
#     def set_pad_val(self, *field_names: str, value=0):
#         raise NotImplementedError
#
#
# class _MultiCollator:
#     """
#     管理所有collator的容器，
#     遵循覆盖原则，后加入的collate_fn会覆盖之前处理的数据。
#     """
#
#     def __init__(self, collate_fns: Union[Callable, List[Callable], None]):
#
#         if collate_fns is None:
#             collate_fns = []
#
#         if isinstance(collate_fns, Callable):
#             collate_fns = [collate_fns]
#
#         self._collators: list = collate_fns
#
#     def __call__(self, ins_lst) -> Dict:
#         out, list_out = {}, []
#         for idx, _collate_fn in enumerate(self._collators):
#             res = _collate_fn(ins_lst)
#             if isinstance(res, Dict):
#                 out.update(res)
#             else:
#                 list_out.append(res)
#             # else:
#             #     raise ValueError(f"the return type of collate_fn {idx} is {type(res)}, but require is dict")
#         if len(out) > 0 and len(list_out) > 0:
#             raise ValueError("the return of collate_fns is not the same, must be dict or list")
#         if len(list_out) == 1:
#             list_out = list_out[-1]
#         # print(list_out)
#         return out if len(out) > 0 else list_out
#
#     def get_collators(self):
#         return self._collators
#
#     def add_collator(self, collator: Callable):
#         self._collators.append(collator)
#
#     def set_as_numpy(self, as_numpy: bool):
#         """
#         存在AutoCollator时，as_numpy控制其返回值的类型
#
#         :param as_numpy:
#         :return:
#         """
#         for collator in self._collators:
#             if isinstance(collator, AutoCollator):
#                 collator.set_as_numpy(as_numpy)
#         return self
#
#     def set_pad_val(self, *field_names, val=0):
#         """
#         存在AutoCollator时，设置field_name的padding值
#
#         :param field_names: 数据集的field名
#         :param val: padding的值
#         :return:
#         """
#         flag = True
#         for collator in self._collators:
#             if isinstance(collator, AutoCollator):
#                 collator.set_pad_val(*field_names, val=val)
#                 flag = False
#         if flag:
#             warnings.warn("AutoCollator is remove, set_padding is unavailable!!")
#         return self
#
#     def set_input(self, *field_names):
#         """
#         设置AutoCollator需要的field_names,未被设置默认过滤掉
#
#         :param field_names:
#         :return:
#         """
#         flag = True
#         for collator in self._collators:
#             if isinstance(collator, AutoCollator):
#                 collator.set_input(*field_names)
#                 flag = False
#         if flag:
#             warnings.warn("AutoCollator is removed, set_input is unavailable!!")
#         return self
#
#
# class AutoCollator(Collator):
#
#     def __init__(self, as_numpy: bool):
#         super(AutoCollator, self).__init__()
#         self.pad_field_value = {}  # field padding 自定义的 padding 值, 默认为0
#         self.need_inputs = set()  # 需要的 field name
#         self.field_dtypes = None  # 每列数据单元的 dtype 类型
#         self.field_dims = None  # 每列数据单元维度
#         self.as_numpy = as_numpy
#
#     def __call__(self, ins_lst: List[Dict]) -> dict:
#         if len(self.need_inputs) == 0:
#             raise ValueError({"set_inputs is None, you should use set_inputs method first!!"})
#         # TODO 这里应该是先 check 有哪些需要 padding，然后check这些是否是可以pad的
#
#         # 第一种情况，设置了 set_input 的值
#         # 第二种情况， 根据数据的类型的判断是否 padding
#         if self.field_dtypes is None and self.field_dims is None:
#             field_dtypes, field_dims = {}, {}
#             for key, value in ins_lst[0].items():
#                 if key in self.need_inputs and self.pad_field_value.get(key, 0) is not None:
#                     field_dtypes[key], field_dims[key] = _get_ele_type_and_dim(value)
#             self.field_dtypes = field_dtypes
#             self.field_dims = field_dims
#
#         pack_ins_lst, pad_ins_lst = {field_name: []
#                                      for field_name in ins_lst[0].keys() if field_name in self.need_inputs}, {}
#         # 将 list 列表内数据按列名打包
#         for per_ins in ins_lst:
#             for field_name, _field_content in per_ins.items():
#                 if field_name in self.need_inputs:
#                     pack_ins_lst[field_name].append(_field_content)
#
#         pad_field_kv = {field_name: 0 for field_name in self.need_inputs}
#         pad_field_kv.update(self.pad_field_value)
#         self.pad_field_value = pad_field_kv
#
#         if len(self.pad_field_value.keys()) > 0:
#             # 去掉不需要 pad 的列，如果 set_input 的列不存在则忽略
#             non_pad_field_names = []
#             for k, v in self.pad_field_value.items():
#                 if v is None:
#                     non_pad_field_names.append(k)
#
#             # drop_field_names = list(set(list(ins_lst[0].keys())) - set(drop_fields))
#             for field_name in non_pad_field_names:
#                 field_array = pack_ins_lst.pop(field_name)
#                 pad_ins_lst[field_name] = np.array(field_array)
#
#             for field_name, field_array in pack_ins_lst.items():
#                 content = pad_content(field_array, field_name, self.field_dtypes[field_name],
#                                       self.field_dims[field_name],
#                                       self.pad_field_value[field_name],
#                                       as_numpy=self.as_numpy)
#                 pad_ins_lst[field_name] = content
#
#         # else:
#         #     # 取出每列的数据，根据类型判断是否能 pad
#         #     for field_name, field_array in pack_ins_lst.items():
#         #         pad_field_array = pad_content(field_array, field_name, self.field_dtypes[field_name],
#         #                                       self.field_dims[field_name],
#         #                                       pad_val=0, as_numpy=self.as_numpy)
#         #         pad_ins_lst[field_name] = pad_field_array
#
#         return pad_ins_lst
#
#     def set_pad_val(self, *field_names, val=0):
#         for field_name in field_names:
#             self.pad_field_value[field_name] = val
#
#     def set_as_numpy(self, as_numpy: bool):
#         self.as_numpy = as_numpy
#
#     def set_input(self, *field_names):
#         for field_name in field_names:
#             self.need_inputs.add(field_name)
#
#
# def pad_content(content, field_name: str, field_type, field_dim: int, pad_val: int, as_numpy: bool):
#
#     if field_type:
#         # 不处理， 返回 np.array 类型
#         if field_dim > 3:
#             return np.array(content)
#         # 元素类型为数值类型 np.int64, np.float64, int, float 等
#         if isinstance(field_type, type) and \
#                 (issubclass(field_type, np.number) or issubclass(field_type, Number)):
#             if field_dim == 0:
#                 array = np.array(content, dtype=field_type)
#             elif field_dim == 1:
#                 max_len = max(map(len, content))
#                 array = np.full((len(content), max_len), pad_val, dtype=field_type)
#                 for i, content_i in enumerate(content):
#                     array[i, :len(content_i)] = content_i
#             elif field_dim == 2:
#                 max_len = max(map(len, content))
#                 max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
#                                     content_i in content])
#                 array = np.full((len(content), max_len, max_word_len), pad_val, dtype=field_type)
#                 for i, content_i in enumerate(content):
#                     for j, content_ii in enumerate(content_i):
#                         array[i, j, :len(content_ii)] = content_ii
#             else:
#                 shape = np.shape(content)
#                 if len(shape) == 4:  # 说明各 dimension 是相同的大小
#                     array = np.array(content, dtype=field_type)
#                 else:
#                     raise RuntimeError(
#                         f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
#             if as_numpy is False:
#                 array = torch.tensor(array)
#             return array
#         # 元素类型为数值类型 torch.float 等
#         elif str(field_type).startswith('torch'):
#             if field_dim == 0:
#                 tensor = torch.tensor(content).to(field_type)
#             elif field_dim == 1:
#                 max_len = max(map(len, content))
#                 tensor = torch.full((len(content), max_len), fill_value=pad_val, dtype=field_type)
#                 for i, content_i in enumerate(content):
#                     tensor[i, :len(content_i)] = content_i.clone().detach()
#             elif field_dim == 2:
#                 max_len = max(map(len, content))
#                 max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
#                                     content_i in content])
#                 tensor = torch.full((len(content), max_len, max_word_len), fill_value=pad_val,
#                                     dtype=field_type)
#                 for i, content_i in enumerate(content):
#                     for j, content_ii in enumerate(content_i):
#                         tensor[i, j, :len(content_ii)] = content_ii.clone().detach()
#             else:
#                 shapes = set([np.shape(content_i) for content_i in content])
#                 if len(shapes) > 1:
#                     raise RuntimeError(
#                         f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
#                 shape = shapes.pop()
#                 if len(shape) == 3:
#                     tensor = torch.full([len(content)] + list(shape), fill_value=pad_val,
#                                         dtype=field_type)
#                     for i, content_i in enumerate(content):
#                         tensor[i] = content_i.clone().detach().to(field_type)
#                 else:
#                     raise RuntimeError(
#                         f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
#             return tensor
#         # TODO 增加jittor/paddle？
#         elif str(field_type).startswith('paddle'):
#             if field_dim == 0:
#                 tensor = paddle.Tensor(content).to(field_type)
#             elif field_dim == 1:
#                 max_len = max(map(len, content))
#                 tensor = paddle.full((len(content), max_len), fill_value=pad_val, dtype=field_type)
#                 for i, content_i in enumerate(content):
#                     tensor[i, :len(content_i)] = content_i.clone().detach()
#             elif field_dim == 2:
#                 max_len = max(map(len, content))
#                 max_word_len = max([max([len(content_ii) for content_ii in content_i]) for
#                                     content_i in content])
#                 tensor = paddle.full((len(content), max_len, max_word_len), fill_value=pad_val,
#                                      dtype=field_type)
#                 for i, content_i in enumerate(content):
#                     for j, content_ii in enumerate(content_i):
#                         tensor[i, j, :len(content_ii)] = content_ii.clone().detach()
#             else:
#                 shapes = set([np.shape(content_i) for content_i in content])
#                 if len(shapes) > 1:
#                     raise RuntimeError(
#                         f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
#                 shape = shapes.pop()
#                 if len(shape) == 3:
#                     tensor = paddle.full([len(content)] + list(shape), fill_value=pad_val,
#                                          dtype=field_type)
#                     for i, content_i in enumerate(content):
#                         tensor[i] = content_i.clone().detach().to(field_type)
#                 else:
#                     raise RuntimeError(
#                         f"Field:{field_name} has 3 dimensions, every sample should have the same shape.")
#             return tensor
#
#         else:
#             return np.array(content)  # 不进行任何操作
#     else:
#         return np.array(content)

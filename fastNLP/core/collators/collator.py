__all__ = [
    'Collator'
]

from typing import List, Union, Dict, Callable, Sequence, Mapping
import os
import sys
import inspect
import re

from fastNLP.core.log import logger
from .padders.get_padder import get_padder
from ...envs import SUPPORT_BACKENDS
from .padders import Padder


from .packer_unpacker import SequencePackerUnpacker, SinglePackerUnpacker, MappingPackerUnpacker, \
    NestedMappingPackerUnpacker

sequence_idx_str = re.compile(r'^_\d+$')  # 形如_0, _1
SUPPORTED_BACKENDS = ['torch', 'jittor', 'paddle', 'oneflow', 'numpy', 'raw', 'auto', None]
# 由于 jittor DataLoader 存在自动的 to_jittor 的转换，所以只需要 collate 成为 numpy 就行
AUTO_BACKEND_MAPPING = {'jittor': 'numpy'}

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
            for backend in SUPPORT_BACKENDS:
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
        return AUTO_BACKEND_MAPPING.get(catch_backend[0], catch_backend[0])

    # 方式 (2)
    for backend in SUPPORT_BACKENDS:
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
    """
    用于 pad 数据的对象。会自动将所有能够 pad （由 fastNLP 根据数据判定能否 pad ）的数据都进行 pad 操作，默认 pad 的值为 0。
    哦安定一个 field 是否可以 pad 的方式为：（1）当前这个 field 是否所有对象都是一样的数据类型；（因此，如果某 field 的数据有些是float
    有些是 int 将知道该 field 被判定为不可 pad 类型。）（2）当前这个 field 是否每个 sample 都具有一样的深度；（因此，例如有个 field 的
    数据转为 batch 类型后为 [1, [1,2]], 会被判定为不可 pad ，因为第一个 sample 与 第二个 sample 深度不同）（3）当前这个 field 的类
    型是否是可以 pad （例如 str 类型的数据）。可以通过设置 logger.setLevel('debug') 来打印是判定不可 pad 的原因。

    .. note::

        ``Collator`` 的原理是使用第一个 ``batch`` 的数据尝试推断每个``field``应该使用哪种类型的 ``Padder``，如果第一个 ``batch``
        的数据刚好比较特殊，可能导致在之后的 pad 中遭遇失败，这种情况请通过 ``set_pad()`` 函数手动设置一下。

    todo 补充 code example 。

    如果需要将某个本可以 pad 的 field 设置为不可 pad ，则可以通过 :meth:`~fastNLP.Collator.set_pad` 的 pad_val 设置为 None 实现。
    如果需要某些 field 不要包含在 pad 之后的结果中，可以使用 :meth:`~fastNLP.Collator.set_ignore` 进行设置。

    Collator 在第一次进行 pad 的时候自动根据设置以及数据情况，为每个 field 获取一个 padder ，在之后的每次调用中，都将使用对应
    的 Padder 给对应的 field 。

    :param backend: 对于可以 pad 的 field，使用哪种 tensor，支持 ['torch','jittor','paddle','oneflow','numpy','raw', auto, None]。
        若为 'auto' ，则在进行 pad 的时候会根据调用的环境决定其 backend 。该参数对不能进行 pad 的数据没用影响，不能 pad
        的数据返回一定是 list 。
    """
    def __init__(self, backend='auto'):
        self.unpack_batch_func = None
        self.pack_batch_func = None
        self.ignore_fields = set()
        self.padders = {}
        self.input_fields = {}
        self.batch_data_type = None  # 只能是 d ，s ，l 三种，分别对应输入的batch的每个sample为 dict, single，list。
        self.set_backend(backend)

    def __call__(self, batch)->Union[List, Dict]:
        """
        batch可能存在三种可能性：List[Dict], List[List], List[Sample]

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
                self.packer_unpacker = SinglePackerUnpacker()  # 不需要做任何调整
            elif self.batch_data_type == 'l':
                self.packer_unpacker = SequencePackerUnpacker()
            elif self.batch_data_type == 'd':
                if any([isinstance(v, Mapping) for v in batch[0].values()]):  # 可能存在 nested 的dict。{'a': {'b': xx}}->{('a', 'b'): value}
                    self.packer_unpacker = NestedMappingPackerUnpacker()
                else:
                    self.packer_unpacker = MappingPackerUnpacker()

        # 将 batch 中各个 field 组成自己的 batch；同时忽略处于 ignore_fields 中的数据。
        unpack_batch = self.packer_unpacker.unpack_batch(batch, self.ignore_fields, self.input_fields)

        pad_batch = {}
        if len(self.padders)==0:  # 第一次运行，准备 padder
            if self.backend == 'auto':  # 如果 backend 为 auto ，则尝试通过调用栈等自动获取 backend 。
                self.backend = _get_backend()

            for field_name, batch_field in unpack_batch.items():
                setting = self.input_fields.get(field_name, {'backend': self.backend, 'pad_val': 0 ,
                                                             'dtype': None, 'pad_fn': None})
                pad_fn = setting['pad_fn']
                if callable(pad_fn):
                    padder = pad_fn
                else:
                    backend = self.backend if setting['backend'] == 'auto' else setting['backend']
                    padder = get_padder(batch_field=batch_field, pad_val=setting['pad_val'],
                                        dtype=setting['dtype'], backend=backend,
                                        field_name=field_name)
                self.padders[field_name] = padder

            if self.batch_data_type == 'l':
                self.padders = dict(sorted(self.padders.items(), key=lambda x:int(x[0][1:])))  # sort, 这样 _0, _1 能够保持顺序
        try:
            for key, padder in self.padders.items():
                batch = unpack_batch.get(key)
                pad_batch[key] = padder(batch)
        except BaseException as e:
            try:
                logger.error(f"The following exception happens when try to pad the `{key}` field with padder:{padder}:")
            except:
                pass
            raise e

        return self.packer_unpacker.pack_batch(pad_batch)  # 根据情况恢复成与输入一致的类型

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
        :param backend: 可选['raw', 'numpy', 'torch', 'paddle', 'jittor', 'oneflow', 'auto']，分别代表，输出为 list, numpy.ndarray,
            torch.Tensor, paddle.Tensor, jittor.Var oneflow.Tensor 类型。若 pad_val 为 None ，该值无意义 。
        :param pad_fn: 指定当前 field 的 pad 函数，传入该函数则 pad_val, dtype, backend 等参数失效。pad_fn 的输入为当前 field 的
            batch 形式。 Collator 将自动 unbatch 数据，然后将各个 field 组成各自的 batch 。pad_func 的输入即为 field 的 batch
            形式，输出将被直接作为结果输出。
        :return: 返回 Collator 自身
        """
        self._renew()

        if self.batch_data_type == 's':
            logger.debug("Set as single field mode.")
            self.input_fields.clear()
        elif self.batch_data_type == 'd':
            if isinstance(field_name, str):
                assert sequence_idx_str.match(field_name) is None, f"Field name:{field_name} will be recognized as list " \
                                                           f"index, but other field is set as dict mode."
        elif self.batch_data_type == 'l':
            if isinstance(field_name, str):
                assert sequence_idx_str.match(field_name) is not None, f"Other field is set as list mode. But the new " \
                                                                       f"field name is {field_name}."

        if field_name == '_single':
            self.batch_data_type = 's'
        elif isinstance(field_name, str) and sequence_idx_str.match(field_name):
            self.batch_data_type = 'l'
        else:
            self.batch_data_type = 'd'

        # 检测是否已经设置了，主要需要考虑它的父亲节点的情况
        ignore_fields = [(field, field) if isinstance(field, tuple) else ((field,), field)
                         for field in self.ignore_fields]
        input_field_names = [(field, field) if isinstance(field, tuple) else ((field,), field)
                             for field in self.input_fields.keys()]
        if isinstance(field_name, tuple):
            _field_name = field_name
        else:
            _field_name = (field_name,)
        for field, o_field in ignore_fields:
            d = _compare_tuple(field, _field_name)
            if d is None:
                continue
            if d == 0:
                logger.rank_zero_warning(f"Field:`{field_name}` has been set as ignored before. It will not be "
                                         f"ignored afterwards.")
                self.ignore_fields.remove(o_field)
            if d > 0:
                raise KeyError(f"Cannot set `{field_name}` as input, since its children `{o_field}` has been set "
                               f"as ignore field.")
            if d < 0:
                raise KeyError(f"Cannot set `{field_name}` as input, since its parent `{o_field}` has been set "
                               f"as ignore field.")
        for field, o_field in input_field_names:
            d = _compare_tuple(field, _field_name)
            if d is None:
                continue
            if d > 0:
                raise KeyError(f"Cannot set `{field_name}` as input, since its children `{o_field}` has been set "
                               f"pad.")
            if d < 0:
                raise KeyError(f"Cannot set `{field_name}` as input, since its parent `{o_field}` has been set "
                               f"pad.")

        if backend is None:
            backend = self.backend
        else:
            assert backend in SUPPORTED_BACKENDS

        self.input_fields[field_name] = {'pad_val': pad_val, 'dtype': dtype, 'backend': backend, 'pad_fn': pad_fn}

        return self

    def set_backend(self, backend:str):
        """
        设置可以 pad 的 field 默认 pad 为什么类型的 tensor

        :param backend: 对于可以 pad 的 field，使用哪种 tensor，支持 ['torch','jittor','paddle','oneflow','numpy','raw', 'auto', None]，
            若为 auto ，则在进行 pad 的时候会自动根据调用的环境决定其 backend 。
        :return:
        """
        assert backend in SUPPORTED_BACKENDS
        self._renew()
        self.backend = backend

    def set_ignore(self, *field_names) -> "Collator":
        """
        如果有的内容不希望输出，可以在此处进行设置，被设置的 field 将在 batch 的输出中被忽略。

            >>> collator = Collator().set_ignore('field1', 'field2')

        :param field_names: 需要忽略的 field 的名称。如果 Dataset 的 __getitem__ 方法返回的是 dict 类型的，则可以直接使用对应的
            field 的 key 来表示，如果是 nested 的 dict，可以使用元组来表示，例如 {'a': {'b': 1}} 中的使用 ('a', 'b'); 如果
            __getitem__ 返回的是 Sequence 类型的，则可以使用 '_0', '_1' 表示序列中第 0 或 1 个元素。
        :return: 返回 Collator 自身
        """
        self._renew()
        input_field_names = [(field, field) if isinstance(field, tuple) else ((field,), field)
                             for field in self.input_fields.keys()]

        # 需要考虑父节点之类的情况
        for field in field_names:
            if not isinstance(field, tuple):
                _field = (field,)
            else:
                _field = field
            for _field_name, o_field_name in input_field_names:
                d = _compare_tuple(_field, _field_name)
                if d is None:
                    continue
                if d == 0:
                    self.input_fields.pop(o_field_name)
                    logger.rank_zero_warning(f"Field:{o_field_name} has been set as pad before. It will be ignored afterwards.")
                if d < 0:
                    self.input_fields.pop(o_field_name)
                    logger.rank_zero_warning(f"Field:{o_field_name} has been set as pad before. It will be ignored afterwards.")
                if d > 0:
                    raise KeyError(f"Cannot ignore {field} since its parent key {o_field_name} has been set as pad.")
            self.ignore_fields.add(field)

        return self

    def _renew(self):
        self.packer_unpacker = None
        self.padders.clear()


def _compare_tuple(t1, t2):
    """
    检测 t1 和 t2 的关系。
    例如 (1, ) 和 (1, ) 关系为 0，表示两者完全没有差异
    例如 (1, ) 和 (2, ) 关系为 None，表示完全不同
    例如 (1, 2, 3) 和 (1, ) 关系为 2，表示前者比后者长 2 位
    但 例如 (1, 2, 3) 和 (2, ) 关系为 None，因为它们从前往后的key 不一样
       例如 (1, 2, 3) 和 (1, 3) 关系为 None，因为它们从前往后的key 不一样

    例如 (1, ) 和 (1, 2, 3) 关系为 -2，表示后者比前者长 2 位
    但 例如 (2, ) 和 (1, 2, 3) 关系为 None，因为它们从前往后的key 不一样
       例如 (1, 3) 和 (1, 2, 3) 关系为 None，因为它们从前往后的key 不一样
    :param t1:
    :param t2:
    :return: None 没有关系; 0 两者完全一样； >0 t1比t2长，<0 t2比t1长
    """
    if t1 == t2:
        return 0
    for _t1, _t2 in zip(t1, t2):  # 会按照最短的计算
        if _t1 != _t2:
            return None
    return len(t1) - len(t2)

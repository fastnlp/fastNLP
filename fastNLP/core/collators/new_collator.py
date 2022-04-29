from typing import List, Union, Dict, Callable, Sequence, Mapping

from fastNLP.core.log import logger
from .padders.get_padder import get_padder

import re

from .utils import unpack_batch_mapping, unpack_batch_nested_mapping, pack_batch_nested_mapping, unpack_batch_sequence, \
    pack_batch_sequence, NESTED_DICT_SEPARATOR

sequence_idx_str = re.compile(r'^_\d+$')  # 形如_0, _1
SUPPORTED_BACKENDS = ['torch', 'jittor', 'paddle', 'numpy', 'raw', None]


class Collator:
    def __init__(self, backend='torch'):
        """
        用于 pad 数据的对象。会自动将所有能够 pad （由 fastNLP 根据数据判定能否 pad ）的数据都进行 pad 操作，默认 pad 的值为 0。
            可使用 set_pad() 函数调整。如果有些 field 不想输出，可以使用 set_ignore() 函数进行设置。

        :param backend: 对于可以 pad 的 field，使用哪种 tensor，支持 ['torch','jittor','paddle','numpy','raw',None]，
            若为 None ，则不进行 padding 。
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
                             f"is {self.batch_data_type}")
            if self.batch_data_type == 's':
                self.unpack_batch_func = lambda x:{'_single': x}  # 不需要做任何调整
                self.pack_batch_func = lambda x:x['_single']
            elif self.batch_data_type == 'l':
                self.unpack_batch_func = unpack_batch_sequence
                self.pack_batch_func = pack_batch_sequence
            elif self.batch_data_type == 'd':
                if any([isinstance(v, Mapping) for v in batch[0].values()]):  # 可能存在 nested 的dict。{'a': {'b': xx}}->{'a@@b': value}
                    self.unpack_batch_func = unpack_batch_nested_mapping
                    self.pack_batch_func = pack_batch_nested_mapping
                else:
                    self.unpack_batch_func = unpack_batch_mapping
                    self.pack_batch_func = lambda x:x

        unpack_batch:Dict = self.unpack_batch_func(batch)  # 将各自 field 组成 batch 形式。

        pad_batch = {}
        if len(self.padders)==0:  # 第一次运行，准备 padder
            for key in unpack_batch.keys():
                if key not in self.input_fields and key not in self.ignore_fields:
                    self.input_fields[key] = {'pad_val': 0, 'dtype': None, 'backend': self.backend}

            for field_name, setting in self.input_fields.items():
                pad_fn = setting.get('pad_fn', None)
                if callable(pad_fn):
                    padder = pad_fn
                else:
                    batch_field = unpack_batch.get(field_name)
                    padder = get_padder(batch_field=batch_field, pad_val=setting['pad_val'],
                                        dtype=setting['dtype'], backend=setting['backend'],
                                        field_name=field_name)
                self.padders[field_name] = padder
            if self.batch_data_type == 'l':
                self.padders = dict(sorted(self.padders.items(), key=lambda x:int(x[0][1:])))  # sort, 这样 _0, _1 能够保持顺序

        for key, padder in self.padders.items():
            batch = unpack_batch.get(key)
            pad_batch[key] = padder(batch)

        return self.pack_batch_func(pad_batch)  # 根据情况恢复成与输入一致的类型

    def set_pad(self, field_name:str, pad_val:Union[int, float, None]=0, dtype=None, backend=None,
                pad_fn:Callable=None) -> "Collator":
        """
        如果需要对某个 field 的内容进行特殊的调整，请使用这个函数。

        :param field_name: 需要调整的 field 的名称。如果 Dataset 的 __getitem__ 方法返回的是 dict 类型的，则可以直接使用对应的
            field 的 key 来表示，如果是 nested 的 dict，可以使用 @@ 来连接不同层次的 key，例如 {'a': {'b': 1}} 中的使用 a@@b;
            如果 __getitem__ 返回的是 Sequence 类型的，则可以使用 '_0', '_1' 表示序列中第 0 或 1 个元素。如果该 field 在数据中没
            有找到，则报错；如果 __getitem__ 返回的是就是整体内容，请使用 "_single" 。
        :param pad_val: 这个 field 的默认 pad 值。如果设置为 None，则表示该 field 不需要 pad , fastNLP 默认只会对可以 pad 的
            field 进行 pad，所以如果对应 field 本身就不是可以 pad 的形式，可以不需要主动设置为 None 。
        :param dtype: 对于需要 pad 的 field ，该 field 的数据 dtype 应该是什么。
        :param backend: 可选[None, 'numpy', 'torch', 'paddle', 'jittor']，分别代表，输出为 list, numpy.ndarray, torch.Tensor,
            paddle.Tensor, jittor.Var 类型。若 pad_val 为 None ，该值只能为 None 或 numpy 。
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
                                                                       f"field name is {field_name}"

        if field_name == '_single':
            self.batch_data_type = 's'
        elif sequence_idx_str.match(field_name):
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

        :param backend: 对于可以 pad 的 field，使用哪种 tensor，支持 ['torch','jittor','paddle','numpy','raw',None]，
            若为 None ，则不进行 padding 。
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
            field 的 key 来表示，如果是 nested 的 dict，可以使用 @@ 来连接不同层次的 key，例如 {'a': {'b': 1}} 中的使用 a@@b;
            如果 __getitem__ 返回的是 Sequence 类型的，则可以使用 '_0', '_1' 表示序列中第 0 或 1 个元素。
        :return: 返回 Collator 自身
        """
        for field_name in field_names:
            if field_name in self.input_fields:
                self.input_fields.pop(field_name)
                logger.warning(f"Field:{field_name} has been set as input before. It will be ignored afterwards.")
            self.padders.pop(field_name, None)  # 如果由的话，将它的 padder 扔掉。
            self.ignore_fields.add(field_name)

        return self



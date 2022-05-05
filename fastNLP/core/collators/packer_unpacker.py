from collections import defaultdict
from functools import reduce
from typing import Sequence, Mapping, Dict


class MappingPackerUnPacker:
    @staticmethod
    def unpack_batch(batch:Sequence[Mapping], ignore_fields:set, input_fields:Dict)->Dict:
        """
        将 Sequence[Mapping] 转为 Dict 。例如 [{'a': [1, 2], 'b': 1}, {'a': [3], 'b': 2}] -> {'a': [[1, 2], [3]], 'b': [1, 2]}

        :param batch:
        :param ignore_fields:
        :param input_fields:
        :return:
        """
        dict_batch = defaultdict(list)
        for sample in batch:
            for key, value in sample.items():
                if key in ignore_fields:
                    continue
                dict_batch[key].append(value)
        return dict_batch

    @staticmethod
    def pack_batch(batch):
        return batch


class NestedMappingPackerUnpacker:
    @staticmethod
    def unpack_batch(batch:Sequence[Mapping], ignore_fields:set, input_fields:Dict):
        """
        将 nested 的 dict 中的内容展开到一个 flat dict 中

        :param batch:
        :param ignore_fields: 需要忽略的 field 。
        :param input_fields: 不需要继续往下衍射的
        :return:
        """
        dict_batch = defaultdict(list)
        for sample in batch:
            for key, value in sample.items():
                if key in ignore_fields:
                    continue
                if isinstance(value, Mapping) and key not in input_fields:
                    _dict_batch = _unpack_batch_nested_mapping(value, ignore_fields, input_fields, _parent=(key,))
                    for key, value in _dict_batch.items():
                        dict_batch[key].append(value)
                else:
                    dict_batch[key].append(value)
        return dict_batch

    @staticmethod
    def pack_batch(batch):
        dicts = []

        for key, value in batch.items():
            if not isinstance(key, tuple):
                key = [key]
            d = {key[-1]: value}
            for k in key[:-1:][::-1]:
                d = {k: d}
            dicts.append(d)
        return reduce(_merge_dict, dicts)


class 


def unpack_batch_nested_mapping(batch:Sequence[Mapping], ignore_fields:set, stop_deep_fields:set)->Dict:
    """
    将 nested 的 dict 中的内容展开到一个 flat dict 中

    :param batch:
    :param ignore_fields: 需要忽略的 field 。
    :param stop_deep_fields: 不需要继续往下衍射的
    :return:
    """
    dict_batch = defaultdict(list)
    for sample in batch:
        for key, value in sample.items():
            if key in ignore_fields:
                continue
            if isinstance(value, Mapping) and key not in stop_deep_fields:
                _dict_batch = _unpack_batch_nested_mapping(value, ignore_fields, stop_deep_fields, _parent=(key,))
                for key, value in _dict_batch.items():
                    dict_batch[key].append(value)
            else:
                dict_batch[key].append(value)
    return dict_batch


def _unpack_batch_nested_mapping(value, ignore_fields, stop_deep_fields, _parent)->Dict:
    _dict = {}
    for k, v in value.items():
        _k = _parent + (k,)
        if _k in ignore_fields:
            continue
        if isinstance(v, Mapping) and _k not in stop_deep_fields:
            __dict = _unpack_batch_nested_mapping(v, ignore_fields, stop_deep_fields, _parent=_k)
            _dict.update(__dict)
        else:
            _dict[_k] = v
    return _dict


def pack_batch_nested_mapping(batch:Mapping) -> Dict:
    """
    需要恢复出 nested 的 dict 原来的样式

    :param batch:
    :return:
    """
    dicts = []

    for key, value in batch.items():
        if not isinstance(key, tuple):
            key = [key]
        d = {key[-1]: value}
        for k in key[:-1:][::-1]:
            d = {k: d}
        dicts.append(d)
    return reduce(_merge_dict, dicts)


def _merge_dict(a, b, path=None):
    "merges b into a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                _merge_dict(a[key], b[key], path + [str(key)])
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a


def unpack_batch_sequence(batch:Sequence[Sequence], ignore_fields)->Dict:
    """
    将 Sequence[Sequence] 转为 Mapping 。例如 [[[1, 2], 2], [[3], 2]] -> {'_0': [[1, 2], [3]], '_1': [1, 2]}

    :param batch:
    :param ignore_fields: 需要忽略的field
    :return:
    """
    dict_batch = defaultdict(list)
    for sample in batch:
        for i, content in enumerate(sample):
            field_name = f'_{i}'
            if field_name in ignore_fields:
                continue
            dict_batch[field_name].append(content)
    return dict_batch


def pack_batch_sequence(batch:Mapping)->Sequence:
    return list(batch.values())
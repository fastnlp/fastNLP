from collections import defaultdict
from functools import reduce
from typing import Sequence, Mapping, Dict

NESTED_DICT_SEPARATOR = '@@'


def unpack_batch_mapping(batch:Sequence[Mapping])->Dict:
    """
    将 Sequence[Mapping] 转为 Dict 。例如 [{'a': [1, 2], 'b': 1}, {'a': [3], 'b': 2}] -> {'a': [[1, 2], [3]], 'b': [1, 2]}

    :param batch:
    :return:
    """
    dict_batch = defaultdict(list)
    for sample in batch:
        for key, value in sample.items():
            dict_batch[key].append(value)
    return dict_batch


def unpack_batch_nested_mapping(batch:Sequence[Mapping], _parent='')->Dict:
    """
    将 nested 的 dict 中的内容展开到一个 flat dict 中

    :param batch:
    :param _parent: 内部使用
    :return:
    """
    dict_batch = defaultdict(list)
    if _parent != '':
        _parent += NESTED_DICT_SEPARATOR
    for sample in batch:
        for key, value in sample.items():
            if isinstance(value, Mapping):
                _dict_batch = _unpack_batch_nested_mapping(value, _parent=_parent + key)
                for key, value in _dict_batch.items():
                    dict_batch[key].append(value)
            else:
                dict_batch[_parent + key].append(value)
    return dict_batch


def _unpack_batch_nested_mapping(value, _parent)->Dict:
    _dict = {}
    _parent += NESTED_DICT_SEPARATOR
    for k, v in value.items():
        if isinstance(v, Mapping):
            __dict = _unpack_batch_nested_mapping(v, _parent=_parent + k)
            _dict.update(__dict)
        else:
            _dict[_parent + k] = v
    return _dict


def pack_batch_nested_mapping(batch:Mapping) -> Dict:
    """
    需要恢复出 nested 的 dict 原来的样式

    :param batch:
    :return:
    """
    dicts = []

    for key, value in batch.items():
        keys = key.split(NESTED_DICT_SEPARATOR)
        d = {keys[-1]: value}
        for key in keys[:-1:][::-1]:
            d = {key: d}
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


def unpack_batch_sequence(batch:Sequence[Sequence])->Dict:
    """
    将 Sequence[Sequence] 转为 Mapping 。例如 [[[1, 2], 2], [[3], 2]] -> {'_0': [[1, 2], [3]], '_1': [1, 2]}

    :param batch:
    :return:
    """
    dict_batch = defaultdict(list)
    for sample in batch:
        for i, content in enumerate(sample):
            dict_batch[f'_{i}'].append(content)
    return dict_batch


def pack_batch_sequence(batch:Mapping)->Sequence:
    return list(batch.values())
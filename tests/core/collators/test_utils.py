
from fastNLP.core.collators.utils import *


def test_unpack_batch_mapping():
    batch = [{'a': [1, 2], 'b': 1}, {'a': [3], 'b': 2}]
    assert unpack_batch_mapping(batch)=={'a': [[1, 2], [3]], 'b': [1, 2]}


def test_unpack_batch_nested_mapping():
    batch = [{'a': [1, 2], 'b': 1, 'c': {'c': 1}}, {'a': [3], 'b': 2, 'c': {'c': 2}}]
    assert unpack_batch_nested_mapping(batch) == {'a': [[1, 2], [3]], 'b': [1, 2], 'c@@c': [1, 2]}

    batch = [{'a': [1, 2], 'b': 1, 'c': {'c': {'c': 1}}}, {'a': [3], 'b': 2, 'c': {'c': {'c': 2}}}]
    assert unpack_batch_nested_mapping(batch) == {'a': [[1, 2], [3]], 'b': [1, 2], 'c@@c@@c': [1, 2]}

    batch = [{'a': [1, 2], 'b': 1, 'c': {'c': {'c': 1, 'd':[1, 1]}, 'd': [1]}},
             {'a': [3], 'b': 2, 'c': {'c': {'c': 2, 'd': [2, 2]}, 'd': [2, 2]}}]
    assert unpack_batch_nested_mapping(batch) == {'a': [[1, 2], [3]], 'b': [1, 2], 'c@@c@@c': [1, 2],
                                                  'c@@c@@d':[[1, 1], [2, 2]], 'c@@d': [[1], [2, 2]]}


def test_pack_batch_nested_mapping():
    batch = {'a': [[1, 2], [3]], 'b': [1, 2], 'c@@c@@c': [1, 2],
             'c@@c@@d':[[1, 1], [2, 2]], 'c@@d': [[1], [2, 2]]}
    new_batch = pack_batch_nested_mapping(batch)
    assert new_batch == {'a': [[1, 2], [3]], 'b': [1, 2],
                         'c': {'c':{'c': [1, 2], 'd': [[1, 1], [2, 2]]}, 'd':[[1], [2, 2]]}}


def test_unpack_batch_sequence():
    batch = [[1, 2, 3], [2, 4, 6]]
    new_batch = unpack_batch_sequence(batch)
    assert new_batch == {'_0': [1, 2], '_1': [2, 4], '_2': [3, 6]}




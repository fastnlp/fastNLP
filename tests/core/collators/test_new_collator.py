
import numpy as np
import pytest

from fastNLP.envs.imports import _NEED_IMPORT_TORCH, _NEED_IMPORT_PADDLE, _NEED_IMPORT_JITTOR

from fastNLP.core.collators.new_collator import Collator


def _assert_equal(d1, d2):
    try:
        if 'torch' in str(type(d1)):
            if 'float64' in str(d2.dtype):
                print(d2.dtype)
            assert (d1 == d2).all().item()
        else:
            assert all(d1 == d2)
    except TypeError:
        assert d1 == d2
    except ValueError:
        assert (d1 == d2).all()


def findDictDiff(d1, d2, path=""):
    for k in d1:
        if k in d2:
            if isinstance(d1[k], dict):
                findDictDiff(d1[k], d2[k], "%s -> %s" % (path, k) if path else k)
            else:
                _assert_equal(d1[k], d2[k])
        else:
            raise RuntimeError("%s%s as key not in d2\n" % ("%s: " % path if path else "", k))


def findListDiff(d1, d2):
    assert len(d1)==len(d2)
    for _d1, _d2 in zip(d1, d2):
        if isinstance(_d1, list):
            findListDiff(_d1, _d2)
        else:
            _assert_equal(_d1, _d2)


class TestCollator:
    def test_run(self):
        dict_batch = [{
            'str': '1',
            'lst_str': ['1'],
            'int': 1,
            'lst_int': [1],
            'nest_lst_int': [[1]],
            'float': 1.1,
            'lst_float': [1.1],
            'bool': True,
            'numpy': np.ones(1),
            'dict': {'1': '1'},
            'set': {'1'},
            'nested_dict': {'a': 1, 'b':[1, 2]}
        },
            {
            'str': '2',
            'lst_str': ['2', '2'],
            'int': 2,
            'lst_int': [1, 2],
            'nest_lst_int': [[1], [1, 2]],
            'float': 2.1,
            'lst_float': [2.1],
            'bool': False,
            'numpy': np.zeros(1),
            'dict': {'1': '2'},
            'set': {'2'},
            'nested_dict': {'a': 2, 'b': [1, 2]}
        }
        ]

        list_batch = [['1', ['1'], 1, [1], [[1]], 1.1, [1.1], True, np.ones(1), {'1': '1'}, {'1'}],
                      ['2', ['2', '2'], 2, [2, 2], [[1], [1, 2]], 2.1, [2.1], False, np.ones(2), {'2': '2'}, {'2'}]]

        raw_pad_batch = {'str': ['1', '2'], 'lst_str': [['1'], ['2', '2']], 'int': [1, 2], 'lst_int': [[1, 0], [1, 2]], 'nest_lst_int': [[[1, 0], [0, 0]], [[1, 0], [1, 2]]], 'float': [1.1, 2.1], 'lst_float': [[1.1], [2.1]], 'bool': [True, False], 'numpy': [np.array([1.]), np.array([0.])], 'dict': {'1': ['1', '2']}, 'set': [{'1'}, {'2'}], 'nested_dict': {'a': [1, 2], 'b': [[1, 2], [1, 2]]}}
        collator = Collator(backend='raw')
        assert raw_pad_batch == collator(dict_batch)
        collator = Collator(backend='raw')
        raw_pad_lst = [['1', '2'], [['1'], ['2', '2']], [1, 2], [[1, 0], [2, 2]], [[[1, 0], [0, 0]], [[1, 0], [1, 2]]],
                       [1.1, 2.1], [[1.1], [2.1]], [True, False], [np.ones(1), np.ones(2)], [{'1': '1'}, {'2': '2'}],
                       [{'1'}, {'2'}]]
        findListDiff(raw_pad_lst, collator(list_batch))

        collator = Collator(backend='numpy')
        numpy_pad_batch = {'str': ['1', '2'], 'lst_str': [['1'], ['2', '2']], 'int': np.array([1, 2]), 'lst_int': np.array([[1, 0], [1, 2]]),
                           'nest_lst_int': np.array([[[1, 0], [0, 0]], [[1, 0], [1, 2]]]), 'float': np.array([1.1, 2.1]),
                           'lst_float': np.array([[1.1], [2.1]]), 'bool': np.array([True, False]), 'numpy': np.array([[1], [0]]),
                           'dict': {'1': ['1', '2']}, 'set': [{'1'}, {'2'}], 'nested_dict': {'a': np.array([1, 2]),
                                                                                             'b': np.array([[1, 2], [1, 2]])}}

        findDictDiff(numpy_pad_batch, collator(dict_batch))
        collator = Collator(backend='numpy')
        numpy_pad_lst = [['1', '2'], [['1'], ['2', '2']], np.array([1, 2]), np.array([[1, 0], [2, 2]]),
                         np.array([[[1, 0], [0, 0]], [[1, 0], [1, 2]]]),
                       np.array([1.1, 2.1]), np.array([[1.1], [2.1]]), np.array([True, False]),
                         np.array([[1, 0], [1, 1]]), [{'1': '1'}, {'2': '2'}],
                       [{'1'}, {'2'}]]
        findListDiff(numpy_pad_lst, collator(list_batch))

        if _NEED_IMPORT_TORCH:
            import torch
            collator = Collator(backend='torch')
            numpy_pad_batch = {'str': ['1', '2'], 'lst_str': [['1'], ['2', '2']], 'int': torch.LongTensor([1, 2]),
                               'lst_int': torch.LongTensor([[1, 0], [1, 2]]),
                               'nest_lst_int': torch.LongTensor([[[1, 0], [0, 0]], [[1, 0], [1, 2]]]),
                               'float': torch.FloatTensor([1.1, 2.1]),
                               'lst_float': torch.FloatTensor([[1.1], [2.1]]), 'bool': torch.BoolTensor([True, False]),
                               'numpy': torch.FloatTensor([[1], [0]]),
                               'dict': {'1': ['1', '2']}, 'set': [{'1'}, {'2'}], 'nested_dict': {'a': torch.LongTensor([1, 2]),
                                                                                                 'b': torch.LongTensor(
                                                                                                     [[1, 2], [1, 2]])}}

            findDictDiff(numpy_pad_batch, collator(dict_batch))
            collator = Collator(backend='torch')
            torch_pad_lst = [['1', '2'], [['1'], ['2', '2']], torch.LongTensor([1, 2]), torch.LongTensor([[1, 0], [2, 2]]),
                             torch.LongTensor([[[1, 0], [0, 0]], [[1, 0], [1, 2]]]),
                             torch.FloatTensor([1.1, 2.1]), torch.FloatTensor([[1.1], [2.1]]), torch.BoolTensor([True, False]),
                             torch.LongTensor([[1, 0], [1, 1]]), [{'1': '1'}, {'2': '2'}],
                             [{'1'}, {'2'}]]
            findListDiff(torch_pad_lst, collator(list_batch))

    def test_pad(self):
        dict_batch = [{
            'str': '1',
            'lst_str': ['1'],
            'int': 1,
            'lst_int': [1],
            'nest_lst_int': [[1]],
            'float': 1.1,
            'lst_float': [1.1],
            'bool': True,
            'numpy': np.ones(1),
            'dict': {'1': '1'},
            'set': {'1'},
            'nested_dict': {'a': 1, 'b':[1, 2]}
        },
            {
                'str': '2',
                'lst_str': ['2', '2'],
                'int': 2,
                'lst_int': [1, 2],
                'nest_lst_int': [[1], [1, 2]],
                'float': 2.1,
                'lst_float': [2.1],
                'bool': False,
                'numpy': np.zeros(1),
                'dict': {'1': '2'},
                'set': {'2'},
                'nested_dict': {'a': 2, 'b': [1, 2]}
            }
        ]

        raw_pad_batch = {'str': ['1', '2'], 'lst_str': [['1'], ['2', '2']], 'int': [1, 2], 'lst_int': [[1, 0], [1, 2]], 'nest_lst_int': [[[1, 0], [0, 0]], [[1, 0], [1, 2]]], 'float': [1.1, 2.1], 'lst_float': [[1.1], [2.1]], 'bool': [True, False], 'numpy': [np.array([1.]), np.array([0.])], 'dict': {'1': ['1', '2']}, 'set': [{'1'}, {'2'}], 'nested_dict': {'a': [1, 2], 'b': [[1, 2], [1, 2]]}}

        # 测试 ignore
        collator = Collator(backend='raw')
        collator.set_ignore('str', 'int', 'lst_int', 'nested_dict@@a')
        raw_pad_batch = {'lst_str': [['1'], ['2', '2']], 'nest_lst_int': [[[1, 0], [0, 0]], [[1, 0], [1, 2]]], 'float': [1.1, 2.1], 'lst_float': [[1.1], [2.1]], 'bool': [True, False], 'numpy': [np.array([1.]), np.array([0.])], 'dict': {'1': ['1', '2']}, 'set': [{'1'}, {'2'}], 'nested_dict': {'b': [[1, 2], [1, 2]]}}
        findDictDiff(raw_pad_batch, collator(dict_batch))

        # 测试 set_pad
        collator = Collator(backend='raw')
        collator.set_pad('str', pad_val=1)
        with pytest.raises(BaseException):
            collator(dict_batch)

        # 测试设置 pad 值
        collator = Collator(backend='raw')
        collator.set_pad('nest_lst_int', pad_val=100)
        collator.set_ignore('str', 'int', 'lst_int', 'nested_dict@@a')
        raw_pad_batch = {'lst_str': [['1'], ['2', '2']], 'nest_lst_int': [[[1, 100], [100, 100]], [[1, 100], [1, 2]]],
                         'float': [1.1, 2.1], 'lst_float': [[1.1], [2.1]], 'bool': [True, False], 'numpy': [np.array([1.]), np.array([0.])], 'dict': {'1': ['1', '2']}, 'set': [{'1'}, {'2'}], 'nested_dict': {'b': [[1, 2], [1, 2]]}}
        findDictDiff(raw_pad_batch, collator(dict_batch))

        # 设置 backend 和 type
        collator.set_pad('float', pad_val=100, backend='numpy', dtype=int)
        raw_pad_batch = {'lst_str': [['1'], ['2', '2']], 'nest_lst_int': [[[1, 100], [100, 100]], [[1, 100], [1, 2]]],
                         'float': np.array([1, 2]), 'lst_float': [[1.1], [2.1]], 'bool': [True, False], 'numpy': [np.array([1.]), np.array([0.])], 'dict': {'1': ['1', '2']}, 'set': [{'1'}, {'2'}], 'nested_dict': {'b': [[1, 2], [1, 2]]}}
        findDictDiff(raw_pad_batch, collator(dict_batch))


        # raw_pad_lst = [['1', '2'], [['1'], ['2', '2']], [1, 2], [[1, 0], [2, 2]], [[[1, 0], [0, 0]], [[1, 0], [1, 2]]],
        #                [1.1, 2.1], [[1.1], [2.1]], [True, False], [np.ones(1), np.ones(2)], [{'1': '1'}, {'2': '2'}],
        #                [{'1'}, {'2'}]]
        list_batch = [['1', ['1'], 1, [1], [[1]], 1.1, [1.1], True, np.ones(1), {'1': '1'}, {'1'}],
                      ['2', ['2', '2'], 2, [2, 2], [[1], [1, 2]], 2.1, [2.1], False, np.ones(2), {'2': '2'}, {'2'}]]
        collator = Collator(backend='raw')
        collator.set_ignore('_0', '_3', '_1')
        collator.set_pad('_4', pad_val=None)
        raw_pad_lst = [[1, 2], [[[1]], [[1], [1, 2]]],
                       [1.1, 2.1], [[1.1], [2.1]], [True, False], [np.ones(1), np.ones(2)], [{'1': '1'}, {'2': '2'}],
                       [{'1'}, {'2'}]]
        findListDiff(raw_pad_lst, collator(list_batch))

        collator = Collator(backend='raw')
        collator.set_pad('_0', pad_val=1)
        with pytest.raises(BaseException):
            collator(dict_batch)

        list_batch = [['1', ['1'], 1, [1], [[1]], 1.1, [1.1], True, np.ones(1), {'1': '1'}, {'1'}],
                      ['2', ['2', '2'], 2, [2, 2], [[1], [1, 2]], 2.1, [2.1], False, np.ones(2), {'2': '2'}, {'2'}]]
        collator = Collator(backend='raw')
        collator.set_ignore('_0', '_3', '_1')
        collator.set_pad('_2', backend='numpy')
        collator.set_pad('_4', backend='numpy', pad_val=100)
        raw_pad_lst = [np.array([1, 2]), np.array([[[1, 100], [100, 100]], [[1, 100], [1, 2]]]),
                       [1.1, 2.1], [[1.1], [2.1]], [True, False], [np.ones(1), np.ones(2)], [{'1': '1'}, {'2': '2'}],
                       [{'1'}, {'2'}]]
        findListDiff(raw_pad_lst, collator(list_batch))

        # _single
        collator = Collator()
        collator.set_pad('_single')
        findListDiff(list_batch, collator(list_batch))








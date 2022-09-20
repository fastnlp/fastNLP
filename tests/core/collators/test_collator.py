
import numpy as np
import pytest

from fastNLP.envs.imports import _NEED_IMPORT_TORCH, _NEED_IMPORT_PADDLE, _NEED_IMPORT_JITTOR, _NEED_IMPORT_ONEFLOW

from fastNLP.core.collators.collator import Collator
from ...helpers.utils import Capturing


def _assert_equal(d1, d2):
    try:
        if 'torch' in str(type(d1)):
            assert (d1 == d2).all().item()
        elif 'oneflow' in str(type(d1)):
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
    @staticmethod
    def setup_class(cls):
        cls.dict_batch = [{
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

        cls.list_batch = [['1', ['1'], 1, [1], [[1]], 1.1, [1.1], True, np.ones(1), {'1': '1'}, {'1'}],
                        ['2', ['2', '2'], 2, [2, 2], [[1], [1, 2]], 2.1, [2.1], False, np.ones(2), {'2': '2'}, {'2'}]]

    def test_run_traw(self):

        raw_pad_batch = {'str': ['1', '2'], 'lst_str': [['1'], ['2', '2']], 'int': [1, 2], 'lst_int': [[1, 0], [1, 2]], 'nest_lst_int': [[[1, 0], [0, 0]], [[1, 0], [1, 2]]], 'float': [1.1, 2.1], 'lst_float': [[1.1], [2.1]], 'bool': [True, False], 'numpy': [np.array([1.]), np.array([0.])], 'dict': {'1': ['1', '2']}, 'set': [{'1'}, {'2'}], 'nested_dict': {'a': [1, 2], 'b': [[1, 2], [1, 2]]}}
        collator = Collator(backend='raw')
        assert raw_pad_batch == collator(self.dict_batch)
        collator = Collator(backend='raw')
        raw_pad_lst = [['1', '2'], [['1'], ['2', '2']], [1, 2], [[1, 0], [2, 2]], [[[1, 0], [0, 0]], [[1, 0], [1, 2]]],
                       [1.1, 2.1], [[1.1], [2.1]], [True, False], [[1, 0], [1, 1]], [{'1': '1'}, {'2': '2'}],
                       [{'1'}, {'2'}]]
        findListDiff(raw_pad_lst, collator(self.list_batch))

    def test_run_numpy(self):

        collator = Collator(backend='numpy')
        numpy_pad_batch = {'str': ['1', '2'], 'lst_str': [['1'], ['2', '2']], 'int': np.array([1, 2]), 'lst_int': np.array([[1, 0], [1, 2]]),
                           'nest_lst_int': np.array([[[1, 0], [0, 0]], [[1, 0], [1, 2]]]), 'float': np.array([1.1, 2.1]),
                           'lst_float': np.array([[1.1], [2.1]]), 'bool': np.array([True, False]), 'numpy': np.array([[1], [0]]),
                           'dict': {'1': ['1', '2']}, 'set': [{'1'}, {'2'}], 'nested_dict': {'a': np.array([1, 2]),
                                                                                             'b': np.array([[1, 2], [1, 2]])}}

        findDictDiff(numpy_pad_batch, collator(self.dict_batch))
        collator = Collator(backend='numpy')
        numpy_pad_lst = [['1', '2'], [['1'], ['2', '2']], np.array([1, 2]), np.array([[1, 0], [2, 2]]),
                         np.array([[[1, 0], [0, 0]], [[1, 0], [1, 2]]]),
                       np.array([1.1, 2.1]), np.array([[1.1], [2.1]]), np.array([True, False]),
                         np.array([[1, 0], [1, 1]]), [{'1': '1'}, {'2': '2'}],
                       [{'1'}, {'2'}]]
        findListDiff(numpy_pad_lst, collator(self.list_batch))

    @pytest.mark.torch
    def test_run_torch(self):
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

        findDictDiff(numpy_pad_batch, collator(self.dict_batch))
        collator = Collator(backend='torch')
        torch_pad_lst = [['1', '2'], [['1'], ['2', '2']], torch.LongTensor([1, 2]), torch.LongTensor([[1, 0], [2, 2]]),
                            torch.LongTensor([[[1, 0], [0, 0]], [[1, 0], [1, 2]]]),
                            torch.FloatTensor([1.1, 2.1]), torch.FloatTensor([[1.1], [2.1]]), torch.BoolTensor([True, False]),
                            torch.LongTensor([[1, 0], [1, 1]]), [{'1': '1'}, {'2': '2'}],
                            [{'1'}, {'2'}]]
        findListDiff(torch_pad_lst, collator(self.list_batch))

    @pytest.mark.oneflow
    def test_run_oneflow(self):
        import oneflow
        collator = Collator(backend='oneflow')
        numpy_pad_batch = {'str': ['1', '2'], 'lst_str': [['1'], ['2', '2']], 'int': oneflow.LongTensor([1, 2]),
                            'lst_int': oneflow.LongTensor([[1, 0], [1, 2]]),
                            'nest_lst_int': oneflow.LongTensor([[[1, 0], [0, 0]], [[1, 0], [1, 2]]]),
                            'float': oneflow.FloatTensor([1.1, 2.1]),
                            'lst_float': oneflow.FloatTensor([[1.1], [2.1]]), 'bool': oneflow.BoolTensor([True, False]),
                            'numpy': oneflow.FloatTensor([[1], [0]]),
                            'dict': {'1': ['1', '2']}, 'set': [{'1'}, {'2'}], 'nested_dict': {'a': oneflow.LongTensor([1, 2]),
                                                                                                'b': oneflow.LongTensor(
                                                                                                    [[1, 2], [1, 2]])}}

        findDictDiff(numpy_pad_batch, collator(self.dict_batch))
        collator = Collator(backend='oneflow')
        oneflow_pad_lst = [['1', '2'], [['1'], ['2', '2']], oneflow.LongTensor([1, 2]), oneflow.LongTensor([[1, 0], [2, 2]]),
                            oneflow.LongTensor([[[1, 0], [0, 0]], [[1, 0], [1, 2]]]),
                            oneflow.FloatTensor([1.1, 2.1]), oneflow.FloatTensor([[1.1], [2.1]]), oneflow.BoolTensor([True, False]),
                            oneflow.LongTensor([[1, 0], [1, 1]]), [{'1': '1'}, {'2': '2'}],
                            [{'1'}, {'2'}]]
        findListDiff(oneflow_pad_lst, collator(self.list_batch))

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
        collator.set_ignore('str', 'int', 'lst_int', ('nested_dict', 'a'))
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
        collator.set_ignore('str', 'int', 'lst_int', ('nested_dict','a'))
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
                       [1.1, 2.1], [[1.1], [2.1]], [True, False], [[1, 0], [1, 1]], [{'1': '1'}, {'2': '2'}],
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
                       [1.1, 2.1], [[1.1], [2.1]], [True, False], [[1, 0], [1, 1]], [{'1': '1'}, {'2': '2'}],
                       [{'1'}, {'2'}]]
        findListDiff(raw_pad_lst, collator(list_batch))

        # _single
        collator = Collator()
        collator.set_pad('_single')
        findListDiff(list_batch, collator(list_batch))

    def test_nest_ignore(self):
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
            'nested_dict': {'int': 1, 'lst_int':[1, 2], 'c': {'int': 1}}
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
                'nested_dict': {'int': 1, 'lst_int': [1, 2], 'c': {'int': 1}}
            }
        ]
        # 测试 ignore
        collator = Collator(backend='raw')
        collator.set_ignore('str', 'int', 'lst_int', ('nested_dict', 'int'))
        raw_pad_batch = {'lst_str': [['1'], ['2', '2']], 'nest_lst_int': [[[1, 0], [0, 0]], [[1, 0], [1, 2]]],
                         'float': [1.1, 2.1], 'lst_float': [[1.1], [2.1]], 'bool': [True, False],
                         'numpy': [np.array([1.]), np.array([0.])], 'dict': {'1': ['1', '2']},
                         'set': [{'1'}, {'2'}], 'nested_dict': {'lst_int': [[1, 2], [1, 2]],
                                                                'c': {'int':[1, 1]}}}
        findDictDiff(raw_pad_batch, collator(dict_batch))

        collator = Collator(backend='raw')
        collator.set_pad(('nested_dict', 'c'), pad_val=None)
        collator.set_ignore('str', 'int', 'lst_int')
        raw_pad_batch = {'lst_str': [['1'], ['2', '2']], 'nest_lst_int': [[[1, 0], [0, 0]], [[1, 0], [1, 2]]],
                         'float': [1.1, 2.1], 'lst_float': [[1.1], [2.1]], 'bool': [True, False],
                         'numpy': [np.array([1.]), np.array([0.])], 'dict': {'1': ['1', '2']},
                         'set': [{'1'}, {'2'}], 'nested_dict': {'lst_int': [[1, 2], [1, 2]],
                                                                'c': [{'int':1}, {'int':1}]}}
        pad_batch = collator(dict_batch)
        findDictDiff(raw_pad_batch, pad_batch)

        collator = Collator(backend='raw')
        collator.set_pad(('nested_dict', 'c'), pad_val=1)
        with pytest.raises(BaseException):
            collator(dict_batch)

        collator = Collator(backend='raw')
        collator.set_ignore('str', 'int', 'lst_int')
        collator.set_pad(('nested_dict', 'c'), pad_fn=lambda x: [d['int'] for d in x])
        pad_batch = collator(dict_batch)
        raw_pad_batch = {'lst_str': [['1'], ['2', '2']], 'nest_lst_int': [[[1, 0], [0, 0]], [[1, 0], [1, 2]]],
                         'float': [1.1, 2.1], 'lst_float': [[1.1], [2.1]], 'bool': [True, False],
                         'numpy': [np.array([1.]), np.array([0.])], 'dict': {'1': ['1', '2']},
                         'set': [{'1'}, {'2'}], 'nested_dict': {'lst_int': [[1, 2], [1, 2]],
                                                                'c': [1, 1]}}
        findDictDiff(raw_pad_batch, pad_batch)

    def test_raise(self, capsys):
        from fastNLP.core.log import logger
        logger.set_stdout('raw')
        # 对于 nested 的情况
        collator = Collator(backend='numpy')
        data = [[1, 2], [2, 3]]
        collator.set_pad('_0')
        collator.set_pad('_0')
        print(collator(data))
        with Capturing() as out:
            collator.set_ignore('_0')
        assert '_0' in out[0]

        data = [{1: {2: 2, 3: 3}}]
        collator = Collator()
        collator.set_pad((1, 2))
        collator.set_pad((1, 3))
        with Capturing() as out:
            collator.set_ignore(1)
        assert '(1, 2)' in out[0] and '(1, 3)' in out[0]
        assert len(collator(data))==0

        collator = Collator()
        collator.set_ignore((1, 2))
        with pytest.raises(KeyError):
            collator.set_pad(1)

        collator = Collator()
        collator.set_ignore(1)
        with pytest.raises(KeyError):
            collator.set_pad((1, 2))

    @pytest.mark.torch
    def test_torch_4d(self):
        collator = Collator(backend='torch')
        data = [{'x': [[[0,1], [2,3]]]}, {'x': [[[0,1]]]}]
        output = collator(data)
        assert output['x'].size() == (2, 1, 2, 2)


@pytest.mark.torch
def test_torch_dl():
    from fastNLP import TorchDataLoader
    from fastNLP import DataSet
    import numpy as np
    import torch

    ds = DataSet({
        'x': [1, 2], 'y': [[1,2], [3]], 'z':[np.ones((1, 2)), np.ones((2, 3))],
        'i': [{'j': [1, 2]}, {'j': [3]}], 'j': ['a', 'b']
    })

    dl = TorchDataLoader(ds, batch_size=2)
    batch = next(iter(dl))
    assert 'x' in batch and 'y' in batch and 'z' in batch and 'i' in batch and 'j' in batch
    assert isinstance(batch['z'], torch.FloatTensor)
    assert isinstance(batch['j'], list)
    assert isinstance(batch['i']['j'], torch.LongTensor)

    dl.set_ignore('x')
    batch = next(iter(dl))
    assert 'x' not in batch and 'y' in batch and 'z' in batch

    dl.set_pad('y', pad_val=None)
    batch = next(iter(dl))
    assert 'x' not in batch and 'y' in batch and 'z' in batch
    assert isinstance(batch['y'], list)
    assert len(batch['y'][0])!=len(batch['y'][1])  # 没有 pad

    dl.set_pad(('i', 'j'), pad_val=None)
    batch = next(iter(dl))
    assert 'x' not in batch and 'y' in batch and 'z' in batch
    assert isinstance(batch['y'], list)
    assert len(batch['y'][0])!=len(batch['y'][1])  # 没有 pad
    assert isinstance(batch['i']['j'], list)
    assert len(batch['i']['j'][0])!=len(batch['i']['j'][1])  # 没有 pad

    with pytest.raises(KeyError):
        dl.set_pad('i', pad_val=None)

@pytest.mark.oneflow
def test_oneflow_dl():
    from fastNLP import OneflowDataLoader
    from fastNLP import DataSet
    import numpy as np
    import oneflow

    ds = DataSet({
        'x': [1, 2], 'y': [[1,2], [3]], 'z':[np.ones((1, 2)), np.ones((2, 3))],
        'i': [{'j': [1, 2]}, {'j': [3]}], 'j': ['a', 'b']
    })

    dl = OneflowDataLoader(ds, batch_size=2)
    batch = next(iter(dl))
    assert 'x' in batch and 'y' in batch and 'z' in batch and 'i' in batch and 'j' in batch
    assert batch['z'].dtype == oneflow.float32
    assert isinstance(batch['j'], list)
    assert batch['i']['j'].dtype, oneflow.long

    dl.set_ignore('x')
    batch = next(iter(dl))
    assert 'x' not in batch and 'y' in batch and 'z' in batch

    dl.set_pad('y', pad_val=None)
    batch = next(iter(dl))
    assert 'x' not in batch and 'y' in batch and 'z' in batch
    assert isinstance(batch['y'], list)
    assert len(batch['y'][0])!=len(batch['y'][1])  # 没有 pad

    dl.set_pad(('i', 'j'), pad_val=None)
    batch = next(iter(dl))
    assert 'x' not in batch and 'y' in batch and 'z' in batch
    assert isinstance(batch['y'], list)
    assert len(batch['y'][0])!=len(batch['y'][1])  # 没有 pad
    assert isinstance(batch['i']['j'], list)
    assert len(batch['i']['j'][0])!=len(batch['i']['j'][1])  # 没有 pad

    with pytest.raises(KeyError):
        dl.set_pad('i', pad_val=None)


def test_compare_tuple():
    from fastNLP.core.collators.collator import _compare_tuple
    for t1, t2, t in zip([(1,), (1, 2, 3), (1,), (1, 2)],
                         [(1, 2, 3), (1,), (2,), (1, 3)],
                         [-2, 2, None, None]):
        assert _compare_tuple(t1, t2) == t

import unittest

import numpy as np
import torch

from fastNLP import DataSetIter, TorchLoaderIter
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import SequentialSampler
from fastNLP import ConcatCollectFn


def generate_fake_dataset(num_samples=1000):
    """
    产生的DataSet包含以下的field {'1':[], '2':[], '3': [], '4':[]}
    :param num_samples: sample的数量
    :return:
    """
    
    max_len = 50
    min_len = 10
    num_features = 4
    
    data_dict = {}
    for i in range(num_features):
        data = []
        lengths = np.random.randint(min_len, max_len, size=(num_samples))
        for length in lengths:
            data.append(np.random.randint(1, 100, size=length))
        data_dict[str(i)] = data
    
    dataset = DataSet(data_dict)
    
    for i in range(num_features):
        if np.random.randint(2) == 0:
            dataset.set_input(str(i))
        else:
            dataset.set_target(str(i))
    return dataset


def construct_dataset(sentences):
    """Construct a data set from a list of sentences.

    :param sentences: list of list of str
    :return dataset: a DataSet object
    """
    dataset = DataSet()
    for sentence in sentences:
        instance = Instance()
        instance['raw_sentence'] = sentence
        dataset.append(instance)
    return dataset


class TestCase1(unittest.TestCase):
    def test_simple(self):
        dataset = construct_dataset(
            [["FastNLP", "is", "the", "most", "beautiful", "tool", "in", "the", "world"] for _ in range(40)])
        dataset.set_target()
        batch = DataSetIter(dataset, batch_size=4, sampler=SequentialSampler(), as_numpy=True)
        
        cnt = 0
        for _, _ in batch:
            cnt += 1
        self.assertEqual(cnt, 10)

    def test_dataset_batching(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        ds.set_input("x")
        ds.set_target("y")
        iter = DataSetIter(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=True)
        for x, y in iter:
            self.assertTrue(isinstance(x["x"], np.ndarray) and isinstance(y["y"], np.ndarray))
            self.assertEqual(len(x["x"]), 4)
            self.assertEqual(len(y["y"]), 4)
            self.assertListEqual(list(x["x"][-1]), [1, 2, 3, 4])
            self.assertListEqual(list(y["y"][-1]), [5, 6])
    
    def test_list_padding(self):
        ds = DataSet({"x": [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]] * 10,
                      "y": [[4, 3, 2, 1], [3, 2, 1], [2, 1], [1]] * 10})
        ds.set_input("x")
        ds.set_target("y")
        iter = DataSetIter(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=True)
        for x, y in iter:
            self.assertEqual(x["x"].shape, (4, 4))
            self.assertEqual(y["y"].shape, (4, 4))
    
    def test_numpy_padding(self):
        ds = DataSet({"x": np.array([[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]] * 10),
                      "y": np.array([[4, 3, 2, 1], [3, 2, 1], [2, 1], [1]] * 10)})
        ds.set_input("x")
        ds.set_target("y")
        iter = DataSetIter(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=True)
        for x, y in iter:
            self.assertEqual(x["x"].shape, (4, 4))
            self.assertEqual(y["y"].shape, (4, 4))
    
    def test_list_to_tensor(self):
        ds = DataSet({"x": [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]] * 10,
                      "y": [[4, 3, 2, 1], [3, 2, 1], [2, 1], [1]] * 10})
        ds.set_input("x")
        ds.set_target("y")
        iter = DataSetIter(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=False)
        for x, y in iter:
            self.assertTrue(isinstance(x["x"], torch.Tensor))
            self.assertEqual(tuple(x["x"].shape), (4, 4))
            self.assertTrue(isinstance(y["y"], torch.Tensor))
            self.assertEqual(tuple(y["y"].shape), (4, 4))
    
    def test_numpy_to_tensor(self):
        ds = DataSet({"x": np.array([[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]] * 10),
                      "y": np.array([[4, 3, 2, 1], [3, 2, 1], [2, 1], [1]] * 10)})
        ds.set_input("x")
        ds.set_target("y")
        iter = DataSetIter(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=False)
        for x, y in iter:
            self.assertTrue(isinstance(x["x"], torch.Tensor))
            self.assertEqual(tuple(x["x"].shape), (4, 4))
            self.assertTrue(isinstance(y["y"], torch.Tensor))
            self.assertEqual(tuple(y["y"].shape), (4, 4))
    
    def test_list_of_list_to_tensor(self):
        ds = DataSet([Instance(x=[1, 2], y=[3, 4]) for _ in range(2)] +
                     [Instance(x=[1, 2, 3, 4], y=[3, 4, 5, 6]) for _ in range(2)])
        ds.set_input("x")
        ds.set_target("y")
        iter = DataSetIter(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=False)
        for x, y in iter:
            self.assertTrue(isinstance(x["x"], torch.Tensor))
            self.assertEqual(tuple(x["x"].shape), (4, 4))
            self.assertTrue(isinstance(y["y"], torch.Tensor))
            self.assertEqual(tuple(y["y"].shape), (4, 4))
    
    def test_list_of_numpy_to_tensor(self):
        ds = DataSet([Instance(x=np.array([1, 2]), y=np.array([3, 4])) for _ in range(2)] +
                     [Instance(x=np.array([1, 2, 3, 4]), y=np.array([3, 4, 5, 6])) for _ in range(2)])
        ds.set_input("x")
        ds.set_target("y")
        iter = DataSetIter(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=False)
        for x, y in iter:
            print(x, y)
    
    def test_sequential_batch(self):
        batch_size = 32
        num_samples = 1000
        dataset = generate_fake_dataset(num_samples)
        
        batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler())
        for batch_x, batch_y in batch:
            pass

    def test_udf_padder(self):
        from fastNLP.core.field import Padder
        alphas = list('abcdefghijk')
        class UDFPadder(Padder):
            def __init__(self):
                super().__init__()

            def __call__(self, contents, field_name, field_ele_dtype, dim):
                results = [alphas[:con] for con in contents]
                return results

        batch_size = 32
        num_samples = 1000
        dataset = generate_fake_dataset(num_samples)
        contents = np.random.randint(5, size=(num_samples))
        dataset.add_field('test', contents, is_input=True, padder=UDFPadder(),
                          ignore_type=True)

        batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler())
        for batch_x, batch_y in batch:
            test = batch_x['test']
            indices = batch.cur_batch_indices
            cons = contents[indices]
            for con,t in zip(cons, test):
                self.assertEqual(alphas[:con], t)

    def test_collect_fn(self):
        batch_size = 32
        num_samples = 1000
        dataset = generate_fake_dataset(num_samples)
        dataset.set_input('1','2')
        dataset.set_target('0','3')

        fn = ConcatCollectFn(inputs=['1', '2'], output='12', pad_val=0, max_len=0, is_input=True, is_target=False)
        dataset.add_collect_fn(fn, name='demo')
        batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler(), drop_last=True)
        for batch_x, batch_y in batch:
            for i in range(batch_size):
                # print(i)
                self.assertEqual(batch_x['12'][i].sum(), batch_x['1'][i].sum() + batch_x['2'][i].sum())
        dataset.delete_collect_fn(name='demo')

        # 测试非input的情况
        dataset.set_input('1', '2', flag=False)  #
        fn = ConcatCollectFn(inputs=['1', '2'], output='12', pad_val=0, max_len=0, is_input=True, is_target=False)
        dataset.add_collect_fn(fn, name='demo')
        batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler(), drop_last=True)
        for batch_x, batch_y in batch:
            for i in range(batch_size):
                self.assertTrue('12' in batch_x)
        dataset.delete_collect_fn(name='demo')
        dataset.set_input('1', '2', flag=True)  #

        # 测试覆盖其它field的情况
        fn = ConcatCollectFn(inputs=['1', '2'], output='3', pad_val=0, max_len=0, is_input=True, is_target=True)
        dataset.add_collect_fn(fn, name='demo')
        batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler(), drop_last=True)
        for batch_x, batch_y in batch:
            for i in range(batch_size):
                # print(i)
                self.assertEqual(batch_y['3'][i].sum(), batch_x['1'][i].sum() + batch_x['2'][i].sum())
        dataset.delete_collect_fn(name='demo')

        # 测试非input，target的情况
        dataset.set_input('1', '2', flag=False)
        fn = ConcatCollectFn(inputs=['1', '2'], output='3', pad_val=0, max_len=0, is_input=True, is_target=True)
        dataset.add_collect_fn(fn, name='demo')
        batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler(), drop_last=True)
        for batch_x, batch_y in batch:
            for i in range(batch_size):
                # print(i)
                self.assertTrue('3' in batch_x)
                self.assertTrue('3' in batch_y)
        dataset.delete_collect_fn(name='demo')

        # 测试加入非法fn的请
        with self.assertRaises(AssertionError):
            dataset.add_collect_fn(1)

        # 测试collect_fn返回值只有一个的情况
        def demo_collect_fn(ins_list):
            return {'3':1}
        dataset.add_collect_fn(demo_collect_fn, name='demo')
        with self.assertRaises(BaseException):
            batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler(), drop_last=True)
            for batch_x, batch_y in batch:
                pass
        dataset.delete_collect_fn(name='demo')

        # 测试多个collect_fn
        dataset.add_collect_fn(demo_collect_fn, name='demo')
        dataset.add_collect_fn(demo_collect_fn, name='demo')
        # 测试删除
        dataset.delete_collect_fn()
        dataset.delete_collect_fn()
        self.assertTrue(dataset.collector.is_empty())

    def test_demo(self):
        import torch

        data = DataSet({
            'x1': [[0, 1],
                   [2]],
            'x2': [[3],
                   [2, 4, 5]
                   ],
            'y': [0, 1]
        })
        data.set_target('y')

        # 所有的collect_fn函数都接受list[(ind1, instance1), (ind2, instance2), ...]作为输入，其中ind1/ind2是该instance在dataset中
        #   的index，instance1/instance2是这次batch取出来的数据，包含了所有的field.
        def concat_collect_fn(ins_list):
            x1 = [ins['x1'] for ind,ins in ins_list]
            x2 = [ins['x2'] for ind,ins in ins_list]
            xs = []
            for i in range(len(ins_list)):
                xs.append(torch.LongTensor(x1[i] + x2[i]))
            # 需要自行pad并转换为tensor，但不需要移动到gpu
            arr = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
            b_x = {'x': arr}
            b_y = {}
            # 返回值一定是两个dict，第一个dict的值会认为是input，第二个dict的值会认为是target. 若名称与已有input或target重复，则
            #   采用返回值。
            return b_x, b_y

        data.add_collect_fn(concat_collect_fn)

        for batch_x, batch_y in DataSetIter(data, sampler=SequentialSampler(), batch_size=2):
            print("batch_x:", batch_x)
            print("batch_y:", batch_y)
            # batch_x: {'x': tensor([[0, 1, 3, 0],
            #                        [2, 2, 4, 5]])}
            # batch_y: {'y': array([0, 1])}

        # 如果取batch过程含有一些参数，可以通过类来实现
        class ConCollectFn:
            def __init__(self, max_len=3):
                self.max_len = max_len
            def __call__(self, ins_list):
                x1 = [ins['x1'] for ind, ins in ins_list]
                x2 = [ins['x2'] for ind, ins in ins_list]
                xs = []
                for i in range(len(ins_list)):
                    xs.append(torch.LongTensor(x1[i] + x2[i])[:self.max_len])
                arr = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
                b_x = {'x': arr}
                b_y = {}
                return b_x, b_y
        data.delete_collect_fn()  # 删除之前的collect_fn
        data.add_collect_fn(ConCollectFn(max_len=3))
        for batch_x, batch_y in DataSetIter(data, sampler=SequentialSampler(), batch_size=2):
            print("batch_x:", batch_x)
            print("batch_y:", batch_y)
            # batch_x: {'x': tensor([[0, 1, 3],
            #                        [2, 2, 4]])}
            # batch_y: {'y': array([0, 1])}

    def testTensorLoaderIter(self):
        class FakeData:
            def __init__(self, return_dict=True):
                self.x = [[1,2,3], [4,5,6]]
                self.return_dict = return_dict

            def __len__(self):
                return len(self.x)

            def __getitem__(self, i):
                x = self.x[i]
                y = 0
                if self.return_dict:
                    return {'x':x}, {'y':y}
                return x, y

        data1 = FakeData()
        dataiter = TorchLoaderIter(data1, batch_size=2)
        for x, y in dataiter:
            print(x, y)

        def func():
            data2 = FakeData(return_dict=False)
            dataiter = TorchLoaderIter(data2, batch_size=2)
        self.assertRaises(Exception, func)

    """
    def test_multi_workers_batch(self):
        batch_size = 32
        pause_seconds = 0.01
        num_samples = 1000
        dataset = generate_fake_dataset(num_samples)

        num_workers = 1
        batch = Batch(dataset, batch_size=batch_size, sampler=SequentialSampler(), num_workers=num_workers)
        for batch_x, batch_y in batch:
            time.sleep(pause_seconds)

        num_workers = 2
        batch = Batch(dataset, batch_size=batch_size, sampler=SequentialSampler(), num_workers=num_workers)
        end1 = time.time()
        for batch_x, batch_y in batch:
            time.sleep(pause_seconds)
    """
    """
    def test_pin_memory(self):
        batch_size = 32
        pause_seconds = 0.01
        num_samples = 1000
        dataset = generate_fake_dataset(num_samples)

        batch = Batch(dataset, batch_size=batch_size, sampler=SequentialSampler(), pin_memory=True)
        # 这里发生OOM
        # for batch_x, batch_y in batch:
        #     time.sleep(pause_seconds)

        num_workers = 2
        batch = Batch(dataset, batch_size=batch_size, sampler=SequentialSampler(), num_workers=num_workers,
                      pin_memory=True)
        # 这里发生OOM
        # for batch_x, batch_y in batch:
        #     time.sleep(pause_seconds)
    """

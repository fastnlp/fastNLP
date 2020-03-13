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

    def test_collect_fn(self):
        batch_size = 32
        num_samples = 1000
        dataset = generate_fake_dataset(num_samples)
        dataset.set_input('1','2')
        dataset.set_target('0','3')

        fn = ConcatCollectFn()
        dataset.add_collect_fn(fn, inputs=['1', '2'],
                               outputs=['12', 'seq_len'],
                               is_input=True, is_target=False)

        batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler(), drop_last=True)
        for batch_x, batch_y in batch:
            for i in range(batch_size):
                # print(i)
                self.assertEqual(batch_x['12'][i].sum(), batch_x['1'][i].sum() + batch_x['2'][i].sum())
                self.assertEqual(
                    batch_x['seq_len'][i],
                    (batch_x['1'][i]!=0).sum() + (batch_x['2'][i]!=0).sum())


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

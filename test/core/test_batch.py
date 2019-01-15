import unittest

import numpy as np
import torch

from fastNLP.core.batch import Batch
from fastNLP.core.dataset import DataSet
from fastNLP.core.dataset import construct_dataset
from fastNLP.core.instance import Instance
from fastNLP.core.sampler import SequentialSampler


class TestCase1(unittest.TestCase):
    def test_simple(self):
        dataset = construct_dataset(
            [["FastNLP", "is", "the", "most", "beautiful", "tool", "in", "the", "world"] for _ in range(40)])
        dataset.set_target()
        batch = Batch(dataset, batch_size=4, sampler=SequentialSampler(), as_numpy=True)

        cnt = 0
        for _, _ in batch:
            cnt += 1
        self.assertEqual(cnt, 10)

    def test_dataset_batching(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        ds.set_input("x")
        ds.set_target("y")
        iter = Batch(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=True)
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
        iter = Batch(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=True)
        for x, y in iter:
            self.assertEqual(x["x"].shape, (4, 4))
            self.assertEqual(y["y"].shape, (4, 4))

    def test_numpy_padding(self):
        ds = DataSet({"x": np.array([[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]] * 10),
                      "y": np.array([[4, 3, 2, 1], [3, 2, 1], [2, 1], [1]] * 10)})
        ds.set_input("x")
        ds.set_target("y")
        iter = Batch(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=True)
        for x, y in iter:
            self.assertEqual(x["x"].shape, (4, 4))
            self.assertEqual(y["y"].shape, (4, 4))

    def test_list_to_tensor(self):
        ds = DataSet({"x": [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]] * 10,
                      "y": [[4, 3, 2, 1], [3, 2, 1], [2, 1], [1]] * 10})
        ds.set_input("x")
        ds.set_target("y")
        iter = Batch(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=False)
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
        iter = Batch(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=False)
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
        iter = Batch(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=False)
        for x, y in iter:
            self.assertTrue(isinstance(x["x"], torch.Tensor))
            self.assertEqual(tuple(x["x"].shape), (4, 4))
            self.assertTrue(isinstance(y["y"], torch.Tensor))
            self.assertEqual(tuple(y["y"].shape), (4, 4))

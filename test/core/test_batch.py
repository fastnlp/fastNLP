import unittest

import numpy as np

from fastNLP.core.batch import Batch
from fastNLP.core.dataset import DataSet
from fastNLP.core.dataset import construct_dataset
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
        ds.set_input(x=True)
        ds.set_target(y=True)
        iter = Batch(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=True)
        for x, y in iter:
            self.assertTrue(isinstance(x["x"], np.ndarray) and isinstance(y["y"], np.ndarray))
            self.assertEqual(len(x["x"]), 4)
            self.assertEqual(len(y["y"]), 4)
            self.assertListEqual(list(x["x"][-1]), [1, 2, 3, 4])
            self.assertListEqual(list(y["y"][-1]), [5, 6])

import unittest

from fastNLP.core.batch import Batch
from fastNLP.core.dataset import construct_dataset
from fastNLP.core.sampler import SequentialSampler


class TestCase1(unittest.TestCase):
    def test_simple(self):
        dataset = construct_dataset(
            [["FastNLP", "is", "the", "most", "beautiful", "tool", "in", "the", "world"] for _ in range(40)])
        dataset.set_target()
        batch = Batch(dataset, batch_size=4, sampler=SequentialSampler(), use_cuda=False)

        cnt = 0
        for _, _ in batch:
            cnt += 1
        self.assertEqual(cnt, 10)

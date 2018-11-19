import unittest

from fastNLP.core.batch import Batch
from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance
from fastNLP.core.sampler import SequentialSampler


class TestCase1(unittest.TestCase):
    def test(self):
        dataset = DataSet([Instance(x=["I", "am", "here"])] * 40)
        batch = Batch(dataset, batch_size=4, sampler=SequentialSampler(), use_cuda=False)

        for batch_x, batch_y in batch:
            print(batch_x, batch_y)

        # TODO: weird due to change in dataset.py

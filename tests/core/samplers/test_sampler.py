import unittest
import random
from fastNLP.core.samplers import SequentialSampler, RandomSampler, BucketSampler
from fastNLP.core.dataset import DataSet
from array import array
import torch

from fastNLP.core.samplers.sampler import ReproduceBatchSampler
from fastNLP.core.drivers.torch_driver.utils import replace_batch_sampler
from tests.helpers.datasets.torch_data import TorchNormalDataset


class SamplerTest(unittest.TestCase):

    def test_sequentialsampler(self):
        ds = DataSet({'x': [1, 2, 3, 4] * 10})
        sqspl = SequentialSampler(ds)
        for idx, inst in enumerate(sqspl):
            self.assertEqual(idx, inst)

    def test_randomsampler(self):
        ds = DataSet({'x': [1, 2, 3, 4] * 10})
        rdspl = RandomSampler(ds)
        ans = [ds[i] for i in rdspl]
        self.assertEqual(len(ans), len(ds))

    def test_bucketsampler(self):
        data_set = DataSet({"x": [[0] * random.randint(1, 10)] * 10, "y": [[5, 6]] * 10})
        sampler = BucketSampler(data_set, num_buckets=3, batch_size=16, seq_len_field_name="seq_len")



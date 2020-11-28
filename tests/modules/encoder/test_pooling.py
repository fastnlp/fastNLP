import unittest

import torch

from fastNLP.modules.encoder.pooling import MaxPool, MaxPoolWithMask, KMaxPool, AvgPool, AvgPoolWithMask


class TestPooling(unittest.TestCase):
    def test_MaxPool(self):
        max_pool_1d = MaxPool(dimension=1)
        x = torch.randn(5, 6, 7)
        self.assertEqual(max_pool_1d(x).size(), (5, 7))

        max_pool_2d = MaxPool(dimension=2)
        self.assertEqual(max_pool_2d(x).size(), (5, 1))

        max_pool_3d = MaxPool(dimension=3)
        x = torch.randn(4, 5, 6, 7)
        self.assertEqual(max_pool_3d(x).size(), (4, 1, 1))

    def test_MaxPoolWithMask(self):
        pool = MaxPoolWithMask()
        x = torch.randn(5, 6, 7)
        mask = (torch.randn(5, 6) > 0).long()
        self.assertEqual(pool(x, mask).size(), (5, 7))

    def test_KMaxPool(self):
        k_pool = KMaxPool(k=3)
        x = torch.randn(4, 5, 6)
        self.assertEqual(k_pool(x).size(), (4, 15))

    def test_AvgPool(self):
        pool = AvgPool()
        x = torch.randn(4, 5, 6)
        self.assertEqual(pool(x).size(), (4, 5))

    def test_AvgPoolWithMask(self):
        pool = AvgPoolWithMask()
        x = torch.randn(5, 6, 7)
        mask = (torch.randn(5, 6) > 0).long()
        self.assertEqual(pool(x, mask).size(), (5, 7))

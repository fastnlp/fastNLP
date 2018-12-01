import unittest

import torch

from fastNLP.core.optimizer import SGD


class TestOptim(unittest.TestCase):
    def test_case(self):
        optim = SGD(torch.LongTensor(10))
        print(optim.__dict__)

        optim_2 = SGD(lr=0.001)
        print(optim_2.__dict__)

        optim_2 = SGD(lr=0.002, momentum=0.989)
        print(optim_2.__dict__)

    def test_case_2(self):
        with self.assertRaises(RuntimeError):
            _ = SGD(0.001)

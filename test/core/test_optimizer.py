import unittest

import torch

from fastNLP.core.optimizer import SGD, Adam


class TestOptim(unittest.TestCase):
    def test_SGD(self):
        optim = SGD(torch.nn.Linear(10, 3).parameters())
        self.assertTrue("lr" in optim.__dict__["settings"])
        self.assertTrue("momentum" in optim.__dict__["settings"])

        optim = SGD(0.001)
        self.assertEqual(optim.__dict__["settings"]["lr"], 0.001)

        optim = SGD(lr=0.001)
        self.assertEqual(optim.__dict__["settings"]["lr"], 0.001)

        optim = SGD(lr=0.002, momentum=0.989)
        self.assertEqual(optim.__dict__["settings"]["lr"], 0.002)
        self.assertEqual(optim.__dict__["settings"]["momentum"], 0.989)

        with self.assertRaises(RuntimeError):
            _ = SGD("???")
        with self.assertRaises(RuntimeError):
            _ = SGD(0.001, lr=0.002)
        with self.assertRaises(RuntimeError):
            _ = SGD(lr=0.009, shit=9000)

    def test_Adam(self):
        optim = Adam(torch.nn.Linear(10, 3).parameters())
        self.assertTrue("lr" in optim.__dict__["settings"])
        self.assertTrue("weight_decay" in optim.__dict__["settings"])

        optim = Adam(0.001)
        self.assertEqual(optim.__dict__["settings"]["lr"], 0.001)

        optim = Adam(lr=0.001)
        self.assertEqual(optim.__dict__["settings"]["lr"], 0.001)

        optim = Adam(lr=0.002, weight_decay=0.989)
        self.assertEqual(optim.__dict__["settings"]["lr"], 0.002)
        self.assertEqual(optim.__dict__["settings"]["weight_decay"], 0.989)

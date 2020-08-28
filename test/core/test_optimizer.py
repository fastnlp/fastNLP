import unittest

import torch

from fastNLP import SGD, Adam, AdamW


class TestOptim(unittest.TestCase):
    def test_SGD(self):
        optim = SGD(model_params=torch.nn.Linear(10, 3).parameters())
        self.assertTrue("lr" in optim.__dict__["settings"])
        self.assertTrue("momentum" in optim.__dict__["settings"])
        res = optim.construct_from_pytorch(torch.nn.Linear(10, 3).parameters())
        self.assertTrue(isinstance(res, torch.optim.SGD))
        
        optim = SGD(lr=0.001)
        self.assertEqual(optim.__dict__["settings"]["lr"], 0.001)
        res = optim.construct_from_pytorch(torch.nn.Linear(10, 3).parameters())
        self.assertTrue(isinstance(res, torch.optim.SGD))
        
        optim = SGD(lr=0.002, momentum=0.989)
        self.assertEqual(optim.__dict__["settings"]["lr"], 0.002)
        self.assertEqual(optim.__dict__["settings"]["momentum"], 0.989)
        
        optim = SGD(0.001)
        self.assertEqual(optim.__dict__["settings"]["lr"], 0.001)
        res = optim.construct_from_pytorch(torch.nn.Linear(10, 3).parameters())
        self.assertTrue(isinstance(res, torch.optim.SGD))
        
        with self.assertRaises(TypeError):
            _ = SGD("???")
        with self.assertRaises(TypeError):
            _ = SGD(0.001, lr=0.002)
    
    def test_Adam(self):
        optim = Adam(model_params=torch.nn.Linear(10, 3).parameters())
        self.assertTrue("lr" in optim.__dict__["settings"])
        self.assertTrue("weight_decay" in optim.__dict__["settings"])
        res = optim.construct_from_pytorch(torch.nn.Linear(10, 3).parameters())
        self.assertTrue(isinstance(res, torch.optim.Adam))
        
        optim = Adam(lr=0.001)
        self.assertEqual(optim.__dict__["settings"]["lr"], 0.001)
        res = optim.construct_from_pytorch(torch.nn.Linear(10, 3).parameters())
        self.assertTrue(isinstance(res, torch.optim.Adam))
        
        optim = Adam(lr=0.002, weight_decay=0.989)
        self.assertEqual(optim.__dict__["settings"]["lr"], 0.002)
        self.assertEqual(optim.__dict__["settings"]["weight_decay"], 0.989)
        
        optim = Adam(0.001)
        self.assertEqual(optim.__dict__["settings"]["lr"], 0.001)
        res = optim.construct_from_pytorch(torch.nn.Linear(10, 3).parameters())
        self.assertTrue(isinstance(res, torch.optim.Adam))

    def test_AdamW(self):
        optim = AdamW(params=torch.nn.Linear(10, 3).parameters())
        self.assertTrue('lr' in optim.defaults)
        self.assertTrue('weight_decay' in optim.defaults)

        optim = AdamW(params=torch.nn.Linear(10, 3).parameters(), lr=0.002, weight_decay=0.989)
        self.assertEqual(optim.defaults['lr'], 0.002)
        self.assertTrue(optim.defaults['weight_decay'], 0.989)

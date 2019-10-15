import unittest

import torch
import torch.nn.functional as F

import fastNLP as loss


class TestLoss(unittest.TestCase):
    def test_CrossEntropyLoss(self):
        ce = loss.CrossEntropyLoss(pred="my_predict", target="my_truth")
        a = torch.randn(3, 5, requires_grad=False)
        b = torch.empty(3, dtype=torch.long).random_(5)
        ans = ce({"my_predict": a}, {"my_truth": b})
        self.assertEqual(ans, torch.nn.functional.cross_entropy(a, b))

        ce = loss.CrossEntropyLoss(pred="my_predict", target="my_truth", class_in_dim=1)
        a = torch.randn(3, 4, 3)
        b = torch.randint(3, (3, 3))
        ans = ce({"my_predict": a}, {"my_truth": b})
        self.assertEqual(ans, torch.nn.functional.cross_entropy(a, b))

        ce = loss.CrossEntropyLoss(pred="my_predict", target="my_truth", class_in_dim=2)
        a = torch.randn(3, 4, 3)
        b = torch.randint(3, (3, 4))
        ans = ce({"my_predict": a}, {"my_truth": b})
        self.assertEqual(ans, torch.nn.functional.cross_entropy(a.transpose(1, 2), b))
    
    def test_BCELoss(self):
        bce = loss.BCELoss(pred="my_predict", target="my_truth")
        a = torch.sigmoid(torch.randn((3, 5), requires_grad=False))
        b = torch.randn((3, 5), requires_grad=False)
        ans = bce({"my_predict": a}, {"my_truth": b})
        self.assertEqual(ans, torch.nn.functional.binary_cross_entropy(a, b))
    
    def test_L1Loss(self):
        l1 = loss.L1Loss(pred="my_predict", target="my_truth")
        a = torch.randn(3, 5, requires_grad=False)
        b = torch.randn(3, 5)
        ans = l1({"my_predict": a}, {"my_truth": b})
        self.assertEqual(ans, torch.nn.functional.l1_loss(a, b))
    
    def test_NLLLoss(self):
        l1 = loss.NLLLoss(pred="my_predict", target="my_truth")
        a = F.log_softmax(torch.randn(3, 5, requires_grad=False), dim=0)
        b = torch.tensor([1, 0, 4])
        ans = l1({"my_predict": a}, {"my_truth": b})
        self.assertEqual(ans, torch.nn.functional.nll_loss(a, b))


class TestLosserError(unittest.TestCase):
    def test_losser1(self):
        # (1) only input, targets passed
        pred_dict = {"pred": torch.zeros(4, 3)}
        target_dict = {'target': torch.zeros(4).long()}
        los = loss.CrossEntropyLoss()
        
        print(los(pred_dict=pred_dict, target_dict=target_dict))
    
    #
    def test_losser2(self):
        # (2) with corrupted size
        pred_dict = {"pred": torch.zeros(16, 3)}
        target_dict = {'target': torch.zeros(16, 3).long()}
        los = loss.CrossEntropyLoss()
        
        with self.assertRaises(RuntimeError):
            print(los(pred_dict=pred_dict, target_dict=target_dict))
    
    def test_losser3(self):
        # (2) with corrupted size
        pred_dict = {"pred": torch.zeros(16, 3), 'stop_fast_param': 0}
        target_dict = {'target': torch.zeros(16).long()}
        los = loss.CrossEntropyLoss()
        
        print(los(pred_dict=pred_dict, target_dict=target_dict))
    
    def test_check_error(self):
        l1 = loss.NLLLoss(pred="my_predict", target="my_truth")
        a = F.log_softmax(torch.randn(3, 5, requires_grad=False), dim=0)
        b = torch.tensor([1, 0, 4])
        with self.assertRaises(Exception):
            ans = l1({"wrong_predict": a, "my": b}, {"my_truth": b})
        
        with self.assertRaises(Exception):
            ans = l1({"my_predict": a}, {"truth": b, "my": a})

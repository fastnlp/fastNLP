import math
import unittest

import torch
import torch as tc
import torch.nn.functional as F

import fastNLP.core.losses as loss


class TestLoss(unittest.TestCase):

    def test_case_1(self):
        loss_func = loss.LossFunc(F.nll_loss)
        nll_loss = loss.NLLLoss()
        y = tc.Tensor(
            [
                [.3, .4, .3],
                [.5, .3, .2],
                [.3, .6, .1],
            ]
        )

        gy = tc.LongTensor(
            [
                0,
                1,
                2,
            ]
        )

        y = tc.log(y)
        los = loss_func({'input': y}, {'target': gy})
        losses = nll_loss({'input': y}, {'target': gy})

        r = -math.log(.3) - math.log(.3) - math.log(.1)
        r /= 3
        print("loss = %f" % (los))
        print("r = %f" % (r))
        print("nll_loss = %f" % (losses))

        self.assertEqual(int(los * 1000), int(r * 1000))

    def test_case_2(self):
        # 验证squash()的正确性

        log = math.log
        loss_func = loss.Loss("nll")

        y = tc.Tensor(
            [
                [[.3, .4, .3], [.3, .4, .3], ],
                [[.5, .3, .2], [.1, .2, .7], ],
                [[.3, .6, .1], [.2, .1, .7], ],
            ]
        )

        gy = tc.LongTensor(
            [
                [0, 2],
                [1, 2],
                [2, 1],
            ]
        )

        y = tc.log(y)
        # los = loss_func({'input': y}, {'target': gy})
        los = loss_func(y, gy)

        r = -log(.3) - log(.3) - log(.1) - log(.3) - log(.7) - log(.1)
        r /= 6

        self.assertEqual(int(los * 1000), int(r * 1000))

    def test_case_3(self):
        # 验证pack_padded_sequence()的正确性
        log = math.log
        loss_func = loss.NLLLoss()
        y = tc.Tensor(
            [
                [[.3, .4, .3], [.3, .2, .5], [.4, .5, .1, ], ],
                [[.5, .3, .2], [.1, .2, .7], [.0, .0, .0, ], ],
                [[.3, .6, .1], [.0, .0, .0], [.0, .0, .0, ], ],
            ]
        )

        gy = tc.LongTensor(
            [
                [0, 2, 1, ],
                [1, 2, 0, ],
                [2, 0, 0, ],
            ]
        )

        lens = [3, 2, 1]

        # pdb.set_trace()

        y = tc.log(y)

        yy = tc.nn.utils.rnn.pack_padded_sequence(y, lens, batch_first=True).data
        gyy = tc.nn.utils.rnn.pack_padded_sequence(gy, lens, batch_first=True).data
        los = loss_func({'input': yy}, {'target': gyy})

        r = -log(.3) - log(.5) - log(.5) - log(.3) - log(.7) - log(.1)
        r /= 6

        self.assertEqual(int(los * 1000), int(r * 1000))

    def test_case_4(self):
        # 验证unpad()的正确性
        log = math.log
        y = tc.Tensor(
            [
                [[.3, .4, .3], [.3, .2, .5], [.4, .5, .1, ], [.6, .3, .1, ], ],
                [[.5, .3, .2], [.1, .2, .7], [.0, .0, .0, ], [.0, .0, .0, ], ],
                [[.3, .6, .1], [.0, .0, .0], [.0, .0, .0, ], [.0, .0, .0, ], ],
            ]
        )

        gy = tc.LongTensor(
            [
                [0, 2, 1, 2, ],
                [1, 2, 0, 0, ],
                [2, 0, 0, 0, ],
            ]
        )

        lens = [4, 2, 1]
        y = tc.log(y)

        loss_func = loss.Loss("nll", pre_pro=["unpad"])
        los = loss_func(y, gy, lens=lens)

        r = -log(.1) - log(.3) - log(.5) - log(.5) - log(.3) - log(.7) - log(.1)
        r /= 7

        self.assertEqual(int(los * 1000), int(r * 1000))

    def test_case_5(self):
        # 验证mask()和make_mask()的正确性
        log = math.log

        y = tc.Tensor(
            [
                [[.5, .3, .2], [.1, .2, .7], [.0, .0, .0, ], [.0, .0, .0, ], ],
                [[.5, .4, .1], [.3, .2, .5], [.4, .5, .1, ], [.6, .1, .3, ], ],
                [[.3, .6, .1], [.3, .2, .5], [.0, .0, .0, ], [.0, .0, .0, ], ],
            ]
        )

        gy = tc.LongTensor(
            [
                [1, 2, 0, 0, ],
                [0, 2, 1, 2, ],
                [2, 1, 0, 0, ],
            ]
        )

        mask = tc.ByteTensor(
            [
                [1, 1, 0, 0, ],
                [1, 1, 1, 1, ],
                [1, 1, 0, 0, ],
            ]
        )

        y = tc.log(y)

        lens = [2, 4, 2]

        loss_func = loss.Loss("nll", pre_pro=["mask"])
        los = loss_func(y, gy, mask=mask)

        los2 = loss_func(y, gy, mask=loss.make_mask(lens, gy.size()[-1]))

        r = -log(.3) - log(.7) - log(.5) - log(.5) - log(.5) - log(.3) - log(.1) - log(.2)
        r /= 8

        self.assertEqual(int(los * 1000), int(r * 1000))
        self.assertEqual(int(los2 * 1000), int(r * 1000))

    def test_case_6(self):
        # 验证unpad_mask()的正确性
        log = math.log
        y = tc.Tensor(
            [
                [[.3, .4, .3], [.3, .2, .5], [.4, .5, .1, ], [.6, .3, .1, ], ],
                [[.5, .3, .2], [.1, .2, .7], [.0, .0, .0, ], [.0, .0, .0, ], ],
                [[.3, .6, .1], [.0, .0, .0], [.0, .0, .0, ], [.0, .0, .0, ], ],
            ]
        )

        gy = tc.LongTensor(
            [
                [0, 2, 1, 2, ],
                [1, 2, 0, 0, ],
                [2, 0, 0, 0, ],
            ]
        )

        lens = [4, 2, 1]

        # pdb.set_trace()

        y = tc.log(y)

        loss_func = loss.Loss("nll", pre_pro=["unpad_mask"])
        los = loss_func(y, gy, lens=lens)

        r = -log(.1) - log(.3) - log(.5) - log(.5) - log(.3) - log(.7) - log(.1)
        r /= 7

        self.assertEqual(int(los * 1000), int(r * 1000))

    def test_case_7(self):
        # 验证一些其他东西
        log = math.log
        y = tc.Tensor(
            [
                [[.3, .4, .3], [.3, .2, .5], [.4, .5, .1, ], [.6, .3, .1, ], ],
                [[.5, .3, .2], [.1, .2, .7], [.0, .0, .0, ], [.0, .0, .0, ], ],
                [[.3, .6, .1], [.0, .0, .0], [.0, .0, .0, ], [.0, .0, .0, ], ],
            ]
        )

        gy = tc.LongTensor(
            [
                [0, 2, 1, 2, ],
                [1, 2, 0, 0, ],
                [2, 0, 0, 0, ],
            ]
        )

        lens = [4, 2, 1]
        y = tc.log(y)

        loss_func = loss.Loss("nll", pre_pro=[], weight=tc.Tensor([1, 1, 0]))
        loss_func.add_pre_pro("unpad_mask")
        los = loss_func(y, gy, lens=lens)

        r = - log(.3) - log(.5) - log(.3)
        r /= 3
        self.assertEqual(int(los * 1000), int(r * 1000))

    def test_case_8(self):
        pass


class TestLoss_v2(unittest.TestCase):
    def test_CrossEntropyLoss(self):
        ce = loss.CrossEntropyLoss(pred="my_predict", target="my_truth")
        a = torch.randn(3, 5, requires_grad=False)
        b = torch.empty(3, dtype=torch.long).random_(5)
        ans = ce({"my_predict": a}, {"my_truth": b})
        self.assertEqual(ans, torch.nn.functional.cross_entropy(a, b))

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

        # print(los(pred_dict=pred_dict, target_dict=target_dict))

    def test_losser3(self):
        # (2) with corrupted size
        pred_dict = {"pred": torch.zeros(16, 3), 'stop_fast_param':0}
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

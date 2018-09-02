
import torch
import numpy as np
import unittest

import fastNLP.modules.utils as utils

class TestUtils(unittest.TestCase):
    def test_case_1(self):
        a = torch.tensor([
            [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]
        ])
        utils.orthogonal(a)

    def test_case_2(self):
        a = np.random.rand(100, 100)
        utils.mst(a)


import unittest

import torch

from fastNLP.models import CNNText
from fastNLP.modules.utils import get_dropout_mask, summary


class TestUtil(unittest.TestCase):
    def test_get_dropout_mask(self):
        tensor = torch.randn(3, 4)
        mask = get_dropout_mask(0.3, tensor)
        self.assertSequenceEqual(mask.size(), torch.Size([3, 4]))
    
    def test_summary(self):
        model = CNNText(embed=(4, 4), num_classes=2, kernel_nums=(9,5), kernel_sizes=(1,3))
        # 4 * 4 + 4 * (9 * 1 + 5 * 3)  +  2 * (9 + 5 + 1) = 142
        self.assertSequenceEqual((142, 142, 0), summary(model))
        model.embed.requires_grad = False
        self.assertSequenceEqual((142, 126, 16), summary(model))

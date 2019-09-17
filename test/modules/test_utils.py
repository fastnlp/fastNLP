import unittest
import torch
from fastNLP.modules.utils import get_dropout_mask

class TestUtil(unittest.TestCase):
    def test_get_dropout_mask(self):
        tensor = torch.randn(3, 4)
        mask = get_dropout_mask(0.3, tensor)
        self.assertSequenceEqual(mask.size(), torch.Size([3, 4]))
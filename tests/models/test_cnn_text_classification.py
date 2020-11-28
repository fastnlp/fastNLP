
import unittest

from .model_runner import *
from fastNLP.models.cnn_text_classification import CNNText


class TestCNNText(unittest.TestCase):
    def init_model(self, kernel_sizes, kernel_nums=(1,3,5)):
        model = CNNText((VOCAB_SIZE, 30),
                        NUM_CLS,
                        kernel_nums=kernel_nums,
                        kernel_sizes=kernel_sizes)
        return model

    def test_case1(self):
        # 测试能否正常运行CNN
        model = self.init_model((1,3,5))
        RUNNER.run_model_with_task(TEXT_CLS, model)

    def test_init_model(self):
        self.assertRaises(Exception, self.init_model, (2,4))
        self.assertRaises(Exception, self.init_model, (2,))

    def test_output(self):
        model = self.init_model((3,), (1,))
        global MAX_LEN
        MAX_LEN = 2
        RUNNER.run_model_with_task(TEXT_CLS, model)

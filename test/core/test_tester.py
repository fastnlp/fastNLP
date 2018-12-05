import unittest

data_name = "pku_training.utf8"
pickle_path = "data_for_tests"


import numpy as np
import torch.nn.functional as F
from torch import nn
import time
from fastNLP.core.utils import CheckError
from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance
from fastNLP.core.losses import BCELoss
from fastNLP.core.losses import CrossEntropyLoss
from fastNLP.core.metrics import AccuracyMetric
from fastNLP.core.optimizer import SGD
from fastNLP.core.tester import Tester
from fastNLP.models.base_model import NaiveClassifier

def prepare_fake_dataset():
    mean = np.array([-3, -3])
    cov = np.array([[1, 0], [0, 1]])
    class_A = np.random.multivariate_normal(mean, cov, size=(1000,))

    mean = np.array([3, 3])
    cov = np.array([[1, 0], [0, 1]])
    class_B = np.random.multivariate_normal(mean, cov, size=(1000,))

    data_set = DataSet([Instance(x=[float(item[0]), float(item[1])], y=[0.0]) for item in class_A] +
                       [Instance(x=[float(item[0]), float(item[1])], y=[1.0]) for item in class_B])
    return data_set


def prepare_fake_dataset2(*args, size=100):
    ys = np.random.randint(4, size=100, dtype=np.int64)
    data = {'y': ys}
    for arg in args:
        data[arg] = np.random.randn(size, 5)
    return DataSet(data=data)

class TestTester(unittest.TestCase):
    def test_case_1(self):
        # 检查报错提示能否正确提醒用户
        # 这里传入多余参数，让其duplicate
        dataset = prepare_fake_dataset2('x1', 'x_unused')
        dataset.rename_field('x_unused', 'x2')
        dataset.set_input('x1', 'x2')
        dataset.set_target('y', 'x1')
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 4)
            def forward(self, x1, x2):
                x1 = self.fc(x1)
                x2 = self.fc(x2)
                x = x1 + x2
                time.sleep(0.1)
                # loss = F.cross_entropy(x, y)
                return {'preds': x}

        model = Model()
        tester = Tester(
            data=dataset,
            model=model,
            metrics=AccuracyMetric())
        tester.test()

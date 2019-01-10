import unittest

import numpy as np

from fastNLP.core.callback import EchoCallback
from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance
from fastNLP.core.losses import BCELoss
from fastNLP.core.optimizer import SGD
from fastNLP.core.trainer import Trainer
from fastNLP.models.base_model import NaiveClassifier


class TestCallback(unittest.TestCase):
    def test_case(self):
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

        data_set = prepare_fake_dataset()
        data_set.set_input("x")
        data_set.set_target("y")

        model = NaiveClassifier(2, 1)

        trainer = Trainer(data_set, model,
                          loss=BCELoss(pred="predict", target="y"),
                          n_epochs=1,
                          batch_size=32,
                          print_every=50,
                          optimizer=SGD(lr=0.1),
                          check_code_level=2,
                          use_tqdm=False,
                          callbacks=[EchoCallback()])
        trainer.train()

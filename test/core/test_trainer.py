import unittest

import numpy as np

from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance
from fastNLP.core.losses import BCELoss
from fastNLP.core.metrics import AccuracyMetric
from fastNLP.core.optimizer import SGD
from fastNLP.core.trainer import Trainer
from fastNLP.models.base_model import NaiveClassifier


class TrainerTestGround(unittest.TestCase):
    def test_case(self):
        mean = np.array([-3, -3])
        cov = np.array([[1, 0], [0, 1]])
        class_A = np.random.multivariate_normal(mean, cov, size=(1000,))

        mean = np.array([3, 3])
        cov = np.array([[1, 0], [0, 1]])
        class_B = np.random.multivariate_normal(mean, cov, size=(1000,))

        data_set = DataSet([Instance(x=[float(item[0]), float(item[1])], y=[0.0]) for item in class_A] +
                           [Instance(x=[float(item[0]), float(item[1])], y=[1.0]) for item in class_B])

        data_set.set_input("x", flag=True)
        data_set.set_target("y", flag=True)

        train_set, dev_set = data_set.split(0.3)

        model = NaiveClassifier(2, 1)

        trainer = Trainer(train_set, model,
                          losser=BCELoss(input="predict", target="y"),
                          metrics=AccuracyMetric(pred="predict", target="y"),
                          n_epochs=10,
                          batch_size=32,
                          print_every=10,
                          validate_every=-1,
                          dev_data=dev_set,
                          optimizer=SGD(0.1),
                          check_code_level=2
                          )
        trainer.train()

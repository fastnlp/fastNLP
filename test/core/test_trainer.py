import os

import torch.nn as nn
import unittest

from fastNLP.core.trainer import SeqLabelTrainer
from fastNLP.core.loss import Loss
from fastNLP.core.optimizer import Optimizer
from fastNLP.models.sequence_modeling import SeqLabeling

class TestTrainer(unittest.TestCase):
    def test_case_1(self):
        args = {"epochs": 3, "batch_size": 8, "validate": True, "use_cuda": True, "pickle_path": "./save/",
                "save_best_dev": True, "model_name": "default_model_name.pkl",
                "loss": Loss(None),
                "optimizer": Optimizer("Adam", lr=0.001, weight_decay=0),
                "vocab_size": 20,
                "word_emb_dim": 100,
                "rnn_hidden_units": 100,
                "num_classes": 3
                }
        trainer = SeqLabelTrainer()
        train_data = [
            [[1, 2, 3, 4, 5, 6], [1, 0, 1, 0, 1, 2]],
            [[2, 3, 4, 5, 1, 6], [0, 1, 0, 1, 0, 2]],
            [[1, 4, 1, 4, 1, 6], [1, 0, 1, 0, 1, 2]],
            [[1, 2, 3, 4, 5, 6], [1, 0, 1, 0, 1, 2]],
            [[2, 3, 4, 5, 1, 6], [0, 1, 0, 1, 0, 2]],
            [[1, 4, 1, 4, 1, 6], [1, 0, 1, 0, 1, 2]],
        ]
        dev_data = train_data
        model = SeqLabeling(args)
        trainer.train(network=model, train_data=train_data, dev_data=dev_data)
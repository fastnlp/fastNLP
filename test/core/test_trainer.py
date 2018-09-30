import os
import unittest

from fastNLP.core.dataset import SeqLabelDataSet
from fastNLP.core.metrics import SeqLabelEvaluator
from fastNLP.core.field import TextField, LabelField
from fastNLP.core.instance import Instance
from fastNLP.core.loss import Loss
from fastNLP.core.optimizer import Optimizer
from fastNLP.core.trainer import SeqLabelTrainer
from fastNLP.models.sequence_modeling import SeqLabeling


class TestTrainer(unittest.TestCase):
    def test_case_1(self):
        args = {"epochs": 3, "batch_size": 2, "validate": False, "use_cuda": False, "pickle_path": "./save/",
                "save_best_dev": True, "model_name": "default_model_name.pkl",
                "loss": Loss("cross_entropy"),
                "optimizer": Optimizer("Adam", lr=0.001, weight_decay=0),
                "vocab_size": 10,
                "word_emb_dim": 100,
                "rnn_hidden_units": 100,
                "num_classes": 5,
                "evaluator": SeqLabelEvaluator()
                }
        trainer = SeqLabelTrainer(**args)

        train_data = [
            [['a', 'b', 'c', 'd', 'e'], ['a', '@', 'c', 'd', 'e']],
            [['a', '@', 'c', 'd', 'e'], ['a', '@', 'c', 'd', 'e']],
            [['a', 'b', '#', 'd', 'e'], ['a', '@', 'c', 'd', 'e']],
            [['a', 'b', 'c', '?', 'e'], ['a', '@', 'c', 'd', 'e']],
            [['a', 'b', 'c', 'd', '$'], ['a', '@', 'c', 'd', 'e']],
            [['!', 'b', 'c', 'd', 'e'], ['a', '@', 'c', 'd', 'e']],
        ]
        vocab = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, '!': 5, '@': 6, '#': 7, '$': 8, '?': 9}
        label_vocab = {'a': 0, '@': 1, 'c': 2, 'd': 3, 'e': 4}

        data_set = SeqLabelDataSet()
        for example in train_data:
            text, label = example[0], example[1]
            x = TextField(text, False)
            x_len = LabelField(len(text), is_target=False)
            y = TextField(label, is_target=False)
            ins = Instance(word_seq=x, truth=y, word_seq_origin_len=x_len)
            data_set.append(ins)

        data_set.index_field("word_seq", vocab)
        data_set.index_field("truth", label_vocab)

        model = SeqLabeling(args)

        trainer.train(network=model, train_data=data_set, dev_data=data_set)
        # If this can run, everything is OK.

        os.system("rm -rf save")
        print("pickle path deleted")

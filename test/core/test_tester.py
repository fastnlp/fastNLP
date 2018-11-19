import os
import unittest

from fastNLP.core.dataset import DataSet
from fastNLP.core.field import TextField, LabelField
from fastNLP.core.instance import Instance
from fastNLP.core.metrics import SeqLabelEvaluator
from fastNLP.core.tester import Tester
from fastNLP.models.sequence_modeling import SeqLabeling

data_name = "pku_training.utf8"
pickle_path = "data_for_tests"


class TestTester(unittest.TestCase):
    def test_case_1(self):
        model_args = {
            "vocab_size": 10,
            "word_emb_dim": 100,
            "rnn_hidden_units": 100,
            "num_classes": 5
        }
        valid_args = {"save_output": True, "validate_in_training": True, "save_dev_input": True,
                      "save_loss": True, "batch_size": 2, "pickle_path": "./save/",
                      "use_cuda": False, "print_every_step": 1, "evaluator": SeqLabelEvaluator()}

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

        data_set = DataSet()
        for example in train_data:
            text, label = example[0], example[1]
            x = TextField(text, False)
            x_len = LabelField(len(text), is_target=False)
            y = TextField(label, is_target=True)
            ins = Instance(word_seq=x, truth=y, word_seq_origin_len=x_len)
            data_set.append(ins)

        data_set.index_field("word_seq", vocab)
        data_set.index_field("truth", label_vocab)

        model = SeqLabeling(model_args)

        tester = Tester(**valid_args)
        tester.test(network=model, dev_data=data_set)
        # If this can run, everything is OK.

        os.system("rm -rf save")
        print("pickle path deleted")

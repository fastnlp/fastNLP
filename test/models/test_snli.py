import unittest
from .model_runner import *
from fastNLP.models.snli import ESIM


class TestSNLIModel(unittest.TestCase):
    def test_snli(self):
        model = ESIM((VOCAB_SIZE, 10), num_labels=NUM_CLS, dropout_rate=0)
        RUNNER.run_model_with_task(NLI, model)

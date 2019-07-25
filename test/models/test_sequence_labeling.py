

import unittest

from .model_runner import *
from fastNLP.models.sequence_labeling import SeqLabeling, AdvSeqLabel
from fastNLP.core.losses import LossInForward

class TesSeqLabel(unittest.TestCase):
    def test_case1(self):
        # 测试能否正常运行CNN
        init_emb = (VOCAB_SIZE, 30)
        model = SeqLabeling(init_emb,
                        hidden_size=30,
                        num_classes=NUM_CLS)

        data = RUNNER.prepare_pos_tagging_data()
        data.set_input('target')
        loss = LossInForward()
        metric = AccuracyMetric(pred=C.OUTPUT, target=C.TARGET, seq_len=C.INPUT_LEN)
        RUNNER.run_model(model, data, loss, metric)


class TesAdvSeqLabel(unittest.TestCase):
    def test_case1(self):
        # 测试能否正常运行CNN
        init_emb = (VOCAB_SIZE, 30)
        model = AdvSeqLabel(init_emb,
                        hidden_size=30,
                        num_classes=NUM_CLS)

        data = RUNNER.prepare_pos_tagging_data()
        data.set_input('target')
        loss = LossInForward()
        metric = AccuracyMetric(pred=C.OUTPUT, target=C.TARGET, seq_len=C.INPUT_LEN)
        RUNNER.run_model(model, data, loss, metric)
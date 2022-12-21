import pytest
from .model_runner import *
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    from fastNLP.models.torch.sequence_labeling import SeqLabeling, AdvSeqLabel, BiLSTMCRF


@pytest.mark.torch
class TestBiLSTM:
    def test_case1(self):
        # 测试能否正常运行CNN
        init_emb = (VOCAB_SIZE, 30)
        model = BiLSTMCRF(init_emb,
                        hidden_size=30,
                        num_classes=NUM_CLS)

        dl = RUNNER.prepare_pos_tagging_data()
        metric = Accuracy()
        RUNNER.run_model(model, dl, metric)


@pytest.mark.torch
class TestSeqLabel:
    def test_case1(self):
        # 测试能否正常运行CNN
        init_emb = (VOCAB_SIZE, 30)
        model = SeqLabeling(init_emb,
                        hidden_size=30,
                        num_classes=NUM_CLS)

        dl = RUNNER.prepare_pos_tagging_data()
        metric = Accuracy()
        RUNNER.run_model(model, dl, metric)


@pytest.mark.torch
class TestAdvSeqLabel:
    def test_case1(self):
        # 测试能否正常运行CNN
        init_emb = (VOCAB_SIZE, 30)
        model = AdvSeqLabel(init_emb,
                        hidden_size=30,
                        num_classes=NUM_CLS)

        dl = RUNNER.prepare_pos_tagging_data()
        metric = Accuracy()
        RUNNER.run_model(model, dl, metric)
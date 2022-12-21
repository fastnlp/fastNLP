import pytest
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch
    from fastNLP.models.torch.biaffine_parser import BiaffineParser
from fastNLP import Metric, seq_len_to_mask
from .model_runner import *


class ParserMetric(Metric):
    r"""
    评估parser的性能

    """

    def __init__(self):
        super().__init__()
        self.num_arc = 0
        self.num_label = 0
        self.num_sample = 0

    def get_metric(self, reset=True):
        res = {'UAS': self.num_arc * 1.0 / self.num_sample, 'LAS': self.num_label * 1.0 / self.num_sample}
        if reset:
            self.num_sample = self.num_label = self.num_arc = 0
        return res

    def update(self, pred1, pred2, target1, target2, seq_len=None):
        r"""

        :param pred1: 边预测logits
        :param pred2: label预测logits
        :param target1: 真实边的标注
        :param target2: 真实类别的标注
        :param seq_len: 序列长度
        :return dict: 评估结果::

            UAS: 不带label时, 边预测的准确率
            LAS: 同时预测边和label的准确率
        """
        if seq_len is None:
            seq_mask = pred1.new_ones(pred1.size(), dtype=torch.long)
        else:
            seq_mask = seq_len_to_mask(seq_len.long()).long()
        # mask out <root> tag
        seq_mask[:, 0] = 0
        head_pred_correct = (pred1 == target1).long() * seq_mask
        label_pred_correct = (pred2 == target2).long() * head_pred_correct
        self.num_arc += head_pred_correct.sum().item()
        self.num_label += label_pred_correct.sum().item()
        self.num_sample += seq_mask.sum().item()


def prepare_parser_data():
    index = 'index'
    ds = DataSet({index: list(range(N_SAMPLES))})
    ds.apply_field(lambda x: RUNNER.gen_var_seq(MAX_LEN, VOCAB_SIZE),
                   field_name=index, new_field_name='words1')
    ds.apply_field(lambda x: RUNNER.gen_seq(len(x), NUM_CLS),
                   field_name='words1', new_field_name='words2')
    # target1 is heads, should in range(0, len(words))
    ds.apply_field(lambda x: RUNNER.gen_seq(len(x), len(x)),
                   field_name='words1', new_field_name='target1')
    ds.apply_field(lambda x: RUNNER.gen_seq(len(x), NUM_CLS),
                   field_name='words1', new_field_name='target2')
    ds.apply_field(len, field_name='words1', new_field_name='seq_len')
    dl = TorchDataLoader(ds, batch_size=BATCH_SIZE)
    return dl


@pytest.mark.torch
class TestBiaffineParser:
    def test_train(self):
        model = BiaffineParser(embed=(VOCAB_SIZE, 10),
                               pos_vocab_size=VOCAB_SIZE, pos_emb_dim=10,
                               rnn_hidden_size=10,
                               arc_mlp_size=10,
                               label_mlp_size=10,
                               num_label=NUM_CLS, encoder='var-lstm')
        ds = prepare_parser_data()
        RUNNER.run_model(model, ds, metrics=ParserMetric())

    def test_train2(self):
        model = BiaffineParser(embed=(VOCAB_SIZE, 10),
                               pos_vocab_size=VOCAB_SIZE, pos_emb_dim=10,
                               rnn_hidden_size=16,
                               arc_mlp_size=10,
                               label_mlp_size=10,
                               num_label=NUM_CLS, encoder='transformer')
        ds = prepare_parser_data()
        RUNNER.run_model(model, ds, metrics=ParserMetric())

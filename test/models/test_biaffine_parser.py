import unittest

import fastNLP
from fastNLP.models.biaffine_parser import BiaffineParser, ParserLoss, ParserMetric
from .model_runner import *


def prepare_parser_data():
    index = 'index'
    ds = DataSet({index: list(range(N_SAMPLES))})
    ds.apply_field(lambda x: RUNNER.gen_var_seq(MAX_LEN, VOCAB_SIZE),
                   field_name=index, new_field_name=C.INPUTS(0),
                   is_input=True)
    ds.apply_field(lambda x: RUNNER.gen_seq(len(x), NUM_CLS),
                   field_name=C.INPUTS(0), new_field_name=C.INPUTS(1),
                   is_input=True)
    # target1 is heads, should in range(0, len(words))
    ds.apply_field(lambda x: RUNNER.gen_seq(len(x), len(x)),
                   field_name=C.INPUTS(0), new_field_name=C.TARGETS(0),
                   is_target=True)
    ds.apply_field(lambda x: RUNNER.gen_seq(len(x), NUM_CLS),
                   field_name=C.INPUTS(0), new_field_name=C.TARGETS(1),
                   is_target=True)
    ds.apply_field(len, field_name=C.INPUTS(0), new_field_name=C.INPUT_LEN,
                   is_input=True, is_target=True)
    return ds


class TestBiaffineParser(unittest.TestCase):
    def test_train(self):
        model = BiaffineParser(init_embed=(VOCAB_SIZE, 10),
                               pos_vocab_size=VOCAB_SIZE, pos_emb_dim=10,
                               rnn_hidden_size=10,
                               arc_mlp_size=10,
                               label_mlp_size=10,
                               num_label=NUM_CLS, encoder='var-lstm')
        ds = prepare_parser_data()
        RUNNER.run_model(model, ds, loss=ParserLoss(), metrics=ParserMetric())

    def test_train2(self):
        model = BiaffineParser(init_embed=(VOCAB_SIZE, 10),
                               pos_vocab_size=VOCAB_SIZE, pos_emb_dim=10,
                               rnn_hidden_size=16,
                               arc_mlp_size=10,
                               label_mlp_size=10,
                               num_label=NUM_CLS, encoder='transformer')
        ds = prepare_parser_data()
        RUNNER.run_model(model, ds, loss=ParserLoss(), metrics=ParserMetric())

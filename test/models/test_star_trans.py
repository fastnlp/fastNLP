from .model_runner import *
from fastNLP.models.star_transformer import STNLICls, STSeqCls, STSeqLabel


# add star-transformer tests, for 3 kinds of tasks.
def test_cls():
    model = STSeqCls((VOCAB_SIZE, 10), NUM_CLS, dropout=0)
    RUNNER.run_model_with_task(TEXT_CLS, model)

def test_nli():
    model = STNLICls((VOCAB_SIZE, 10), NUM_CLS, dropout=0)
    RUNNER.run_model_with_task(NLI, model)

def test_seq_label():
    model = STSeqLabel((VOCAB_SIZE, 10), NUM_CLS, dropout=0)
    RUNNER.run_model_with_task(POS_TAGGING, model)

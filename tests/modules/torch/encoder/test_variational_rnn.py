import pytest

import numpy as np
from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    import torch
    from fastNLP.modules.torch.encoder.variational_rnn import VarLSTM


@pytest.mark.torch
class TestMaskedRnn:
    def test_case_1(self):
        masked_rnn = VarLSTM(input_size=1, hidden_size=1, bidirectional=True, batch_first=True)
        x = torch.tensor([[[1.0], [2.0]]])
        print(x.size())
        y = masked_rnn(x)

    def test_case_2(self):
        input_size = 12
        batch = 16
        hidden = 10
        masked_rnn = VarLSTM(input_size=input_size, hidden_size=hidden, bidirectional=False, batch_first=True)

        xx = torch.randn((batch, 32, input_size))
        y, _ = masked_rnn(xx)
        assert(tuple(y.shape) == (batch, 32, hidden))


import torch
import unittest

from fastNLP.modules.encoder.variational_rnn import VarMaskedFastLSTM

class TestMaskedRnn(unittest.TestCase):
    def test_case_1(self):
        masked_rnn = VarMaskedFastLSTM(input_size=1, hidden_size=1, bidirectional=True, batch_first=True)
        x = torch.tensor([[[1.0], [2.0]]])
        print(x.size())
        y = masked_rnn(x)
        mask = torch.tensor([[[1], [1]]])
        y = masked_rnn(x, mask=mask)
        mask = torch.tensor([[[1], [0]]])
        y = masked_rnn(x, mask=mask)

    def test_case_2(self):
        masked_rnn = VarMaskedFastLSTM(input_size=1, hidden_size=1, bidirectional=False, batch_first=True)
        x = torch.tensor([[[1.0], [2.0]]])
        print(x.size())
        y = masked_rnn(x)
        mask = torch.tensor([[[1], [1]]])
        y = masked_rnn(x, mask=mask)
        xx = torch.tensor([[[1.0]]])
        #y, hidden = masked_rnn.step(xx)
        #step() still has a bug
        #y, hidden = masked_rnn.step(xx, mask=mask)
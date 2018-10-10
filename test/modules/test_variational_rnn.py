import unittest

import numpy as np
import torch

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
        input_size = 12
        batch = 16
        hidden = 10
        masked_rnn = VarMaskedFastLSTM(input_size=input_size, hidden_size=hidden, bidirectional=False, batch_first=True)

        x = torch.randn((batch, input_size))
        output, _ = masked_rnn.step(x)
        self.assertEqual(tuple(output.shape), (batch, hidden))

        xx = torch.randn((batch, 32, input_size))
        y, _ = masked_rnn(xx)
        self.assertEqual(tuple(y.shape), (batch, 32, hidden))

        xx = torch.randn((batch, 32, input_size))
        mask = torch.from_numpy(np.random.randint(0, 2, size=(batch, 32))).to(xx)
        y, _ = masked_rnn(xx, mask=mask)
        self.assertEqual(tuple(y.shape), (batch, 32, hidden))

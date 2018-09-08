import unittest

from fastNLP.core.action import Action, Batchifier, SequentialSampler


class TestAction(unittest.TestCase):
    def test_case_1(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8]
        y = [1, 1, 1, 1, 2, 2, 2, 2]
        data = []
        for i in range(len(x)):
            data.append([[x[i]], [y[i]]])
        data = Batchifier(SequentialSampler(data), batch_size=2, drop_last=False)
        action = Action()
        for batch_x in action.make_batch(data, use_cuda=False, output_length=True, max_len=None):
            print(batch_x)


import _pickle

import numpy as np
import torch

from fastNLP.action.action import Action
from fastNLP.action.action import RandomSampler, Batchifier
from fastNLP.modules.utils import seq_mask


class BaseTester(Action):
    """docstring for Tester"""

    def __init__(self, test_args):
        """
        :param test_args: named tuple
        """
        super(BaseTester, self).__init__()
        self.validate_in_training = test_args["validate_in_training"]
        self.save_dev_data = None
        self.save_output = test_args["save_output"]
        self.output = None
        self.save_loss = test_args["save_loss"]
        self.mean_loss = None
        self.batch_size = test_args["batch_size"]
        self.pickle_path = test_args["pickle_path"]
        self.iterator = None

        self.model = None
        self.eval_history = []

    def test(self, network):
        # print("--------------testing----------------")
        self.model = network

        # turn on the testing mode; clean up the history
        self.mode(network, test=True)

        dev_data = self.prepare_input(self.pickle_path)

        self.iterator = iter(Batchifier(RandomSampler(dev_data), self.batch_size, drop_last=True))

        batch_output = list()
        num_iter = len(dev_data) // self.batch_size

        for step in range(num_iter):
            batch_x, batch_y = self.batchify(dev_data)

            prediction = self.data_forward(network, batch_x)
            eval_results = self.evaluate(prediction, batch_y)

            if self.save_output:
                batch_output.append(prediction)
            if self.save_loss:
                self.eval_history.append(eval_results)

    def prepare_input(self, data_path):
        """
        Save the dev data once it is loaded. Can return directly next time.
        :param data_path: str, the path to the pickle data for dev
        :return save_dev_data: list. Each entry is a sample, which is also a list of features and label(s).
        """
        if self.save_dev_data is None:
            data_dev = _pickle.load(open(data_path + "/data_train.pkl", "rb"))
            self.save_dev_data = data_dev
        return self.save_dev_data

    def batchify(self, data):
        """
        1. Perform batching from data and produce a batch of training data.
        2. Add padding.
        :param data: list. Each entry is a sample, which is also a list of features and label(s).
            E.g.
                [
                    [[word_11, word_12, word_13], [label_11. label_12]],  # sample 1
                    [[word_21, word_22, word_23], [label_21. label_22]],  # sample 2
                    ...
                ]
        :return batch_x: list. Each entry is a list of features of a sample. [batch_size, max_len]
                 batch_y: list. Each entry is a list of labels of a sample.  [batch_size, num_labels]
        """
        indices = next(self.iterator)
        batch = [data[idx] for idx in indices]
        batch_x = [sample[0] for sample in batch]
        batch_y = [sample[1] for sample in batch]
        batch_x = self.pad(batch_x)
        return batch_x, batch_y

    @staticmethod
    def pad(batch, fill=0):
        """
        Pad a batch of samples to maximum length.
        :param batch: list of list
        :param fill: word index to pad, default 0.
        :return: a padded batch
        """
        max_length = max([len(x) for x in batch])
        for idx, sample in enumerate(batch):
            if len(sample) < max_length:
                batch[idx] = sample + [fill * (max_length - len(sample))]
        return batch

    def data_forward(self, network, data):
        raise NotImplementedError

    def evaluate(self, predict, truth):
        raise NotImplementedError

    @property
    def matrices(self):
        raise NotImplementedError

    def mode(self, model, test=True):
        """To do: combine this function with Trainer ?? """
        if test:
            model.eval()
        else:
            model.train()
        self.eval_history.clear()


class POSTester(BaseTester):
    """
    Tester for sequence labeling.
    """

    def __init__(self, test_args):
        super(POSTester, self).__init__(test_args)
        self.max_len = None
        self.mask = None
        self.batch_result = None

    def data_forward(self, network, x):
        """To Do: combine with Trainer

        :param network: the PyTorch model
        :param x: list of list, [batch_size, max_len]
        :return y: [batch_size, num_classes]
        """
        seq_len = [len(seq) for seq in x]
        x = torch.Tensor(x).long()
        self.batch_size = x.size(0)
        self.max_len = x.size(1)
        self.mask = seq_mask(seq_len, self.max_len)
        y = network(x)
        return y

    def evaluate(self, predict, truth):
        truth = torch.Tensor(truth)
        loss, prediction = self.model.loss(predict, truth, self.mask, self.batch_size, self.max_len)
        return loss.data

    def matrices(self):
        return np.mean(self.eval_history)

import _pickle
import os

import numpy as np
import torch

from fastNLP.core.action import Action
from fastNLP.core.action import RandomSampler, Batchifier
from fastNLP.modules import utils


class BaseTester(Action):
    """docstring for Tester"""

    def __init__(self, test_args, action=None):
        """
        :param test_args: a dict-like object that has __getitem__ method, can be accessed by "test_args["key_str"]"
        """
        super(BaseTester, self).__init__()
        self.action = action if action is not None else Action()
        self.validate_in_training = test_args["validate_in_training"]
        self.save_dev_data = None
        self.save_output = test_args["save_output"]
        self.output = None
        self.save_loss = test_args["save_loss"]
        self.mean_loss = None
        self.batch_size = test_args["batch_size"]
        self.pickle_path = test_args["pickle_path"]
        self.iterator = None
        self.use_cuda = test_args["use_cuda"]

        self.model = None
        self.eval_history = []
        self.batch_output = []

    def test(self, network):
        if torch.cuda.is_available() and self.use_cuda:
            self.model = network.cuda()
        else:
            self.model = network

        # turn on the testing mode; clean up the history
        self.action.mode(network, test=True)
        self.eval_history.clear()
        self.batch_output.clear()

        dev_data = self.prepare_input(self.pickle_path)

        iterator = iter(Batchifier(RandomSampler(dev_data), self.batch_size, drop_last=True))

        num_iter = len(dev_data) // self.batch_size

        for step in range(num_iter):
            batch_x, batch_y = self.action.make_batch(iterator, dev_data)

            prediction = self.data_forward(network, batch_x)

            eval_results = self.evaluate(prediction, batch_y)

            if self.save_output:
                self.batch_output.append(prediction)
            if self.save_loss:
                self.eval_history.append(eval_results)

    def prepare_input(self, data_path):
        """
        Save the dev data once it is loaded. Can return directly next time.
        :param data_path: str, the path to the pickle data for dev
        :return save_dev_data: list. Each entry is a sample, which is also a list of features and label(s).
        """
        if self.save_dev_data is None:
            data_dev = _pickle.load(open(data_path + "data_dev.pkl", "rb"))
            self.save_dev_data = data_dev
        return self.save_dev_data

    def data_forward(self, network, x):
        raise NotImplementedError

    def evaluate(self, predict, truth):
        raise NotImplementedError

    @property
    def metrics(self):
        raise NotImplementedError

    def show_matrices(self):
        """
        This is called by Trainer to print evaluation on dev set.
        :return print_str: str
        """
        raise NotImplementedError


class POSTester(BaseTester):
    """
    Tester for sequence labeling.
    """

    def __init__(self, test_args, action=None):
        """
        :param test_args: a dict-like object that has __getitem__ method, can be accessed by "test_args["key_str"]"
        """
        super(POSTester, self).__init__(test_args, action)
        self.max_len = None
        self.mask = None
        self.batch_result = None

    def data_forward(self, network, inputs):
        if not isinstance(inputs, tuple):
            raise RuntimeError("[fastnlp] output_length must be true for sequence modeling.")
        # unpack the returned value from make_batch
        x, seq_len = inputs[0], inputs[1]
        x = torch.Tensor(x).long()
        batch_size, max_len = x.size(0), x.size(1)
        mask = utils.seq_mask(seq_len, max_len)
        mask = mask.byte().view(batch_size, max_len)

        if torch.cuda.is_available() and self.use_cuda:
            x = x.cuda()
            mask = mask.cuda()
        self.mask = mask

        y = network(x)
        return y

    def evaluate(self, predict, truth):
        truth = torch.Tensor(truth)
        if torch.cuda.is_available() and self.use_cuda:
            truth = truth.cuda()
        batch_size, max_len = predict.size(0), predict.size(1)
        loss = self.model.loss(predict, truth, self.mask) / batch_size

        prediction = self.model.prediction(predict, self.mask)
        results = torch.Tensor(prediction).view(-1,)
        # make sure "results" is in the same device as "truth"
        results = results.to(truth)
        accuracy = torch.sum(results == truth.view((-1,))) / results.shape[0]
        return [loss.data, accuracy.data]

    def metrics(self):
        batch_loss = np.mean([x[0] for x in self.eval_history])
        batch_accuracy = np.mean([x[1] for x in self.eval_history])
        return batch_loss, batch_accuracy

    def show_matrices(self):
        """
        This is called by Trainer to print evaluation on dev set.
        :return print_str: str
        """
        loss, accuracy = self.metrics()
        return "dev loss={:.2f}, accuracy={:.2f}".format(loss, accuracy)


class ClassTester(BaseTester):
    """Tester for classification."""

    def __init__(self, test_args):
        """
        :param test_args: a dict-like object that has __getitem__ method, \
            can be accessed by "test_args["key_str"]"
        """
        # super(ClassTester, self).__init__()
        self.pickle_path = test_args["pickle_path"]

        self.save_dev_data = None
        self.output = None
        self.mean_loss = None
        self.iterator = None

        if "test_name" in test_args:
            self.test_name = test_args["test_name"]
        else:
            self.test_name = "data_test.pkl"

        if "validate_in_training" in test_args:
            self.validate_in_training = test_args["validate_in_training"]
        else:
            self.validate_in_training = False

        if "save_output" in test_args:
            self.save_output = test_args["save_output"]
        else:
            self.save_output = False

        if "save_loss" in test_args:
            self.save_loss = test_args["save_loss"]
        else:
            self.save_loss = True

        if "batch_size" in test_args:
            self.batch_size = test_args["batch_size"]
        else:
            self.batch_size = 50
        if "use_cuda" in test_args:
            self.use_cuda = test_args["use_cuda"]
        else:
            self.use_cuda = True

        if "max_len" in test_args:
            self.max_len = test_args["max_len"]
        else:
            self.max_len = None

        self.model = None
        self.eval_history = []
        self.batch_output = []

    def test(self, network):
        # prepare model
        if torch.cuda.is_available() and self.use_cuda:
            self.model = network.cuda()
        else:
            self.model = network

        # no backward setting for model
        for param in self.model.parameters():
            param.requires_grad = False

        # turn on the testing mode; clean up the history
        self.mode(network, test=True)

        # prepare test data
        data_test = self.prepare_input(self.pickle_path, self.test_name)

        # data generator
        self.iterator = iter(Batchifier(
            RandomSampler(data_test), self.batch_size, drop_last=False))

        # test
        n_batches = len(data_test) // self.batch_size
        n_print = n_batches // 10
        step = 0
        for batch_x, batch_y in self.make_batch(data_test, max_len=self.max_len):
            prediction = self.data_forward(network, batch_x)
            eval_results = self.evaluate(prediction, batch_y)

            if self.save_output:
                self.batch_output.append(prediction)
            if self.save_loss:
                self.eval_history.append(eval_results)

            if step % n_print == 0:
                print("step: {:>5}".format(step))

            step += 1

    def prepare_input(self, data_dir, file_name):
        """Prepare data."""
        file_path = os.path.join(data_dir, file_name)
        with open(file_path, 'rb') as f:
            data = _pickle.load(f)
        return data

    def make_batch(self, data, max_len=None):
        """Batch and pad data."""
        for indices in self.iterator:
            # generate batch and pad
            batch = [data[idx] for idx in indices]
            batch_x = [sample[0] for sample in batch]
            batch_y = [sample[1] for sample in batch]
            batch_x = self.pad(batch_x)

            # convert to tensor
            batch_x = torch.tensor(batch_x, dtype=torch.long)
            batch_y = torch.tensor(batch_y, dtype=torch.long)
            if torch.cuda.is_available() and self.use_cuda:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

            # trim data to max_len
            if max_len is not None and batch_x.size(1) > max_len:
                batch_x = batch_x[:, :max_len]

            yield batch_x, batch_y

    def data_forward(self, network, x):
        """Forward through network."""
        logits = network(x)
        return logits

    def evaluate(self, y_logit, y_true):
        """Return y_pred and y_true."""
        y_prob = torch.nn.functional.softmax(y_logit, dim=-1)
        return [y_prob, y_true]

    def metrics(self):
        """Compute accuracy."""
        y_prob, y_true = zip(*self.eval_history)
        y_prob = torch.cat(y_prob, dim=0)
        y_pred = torch.argmax(y_prob, dim=-1)
        y_true = torch.cat(y_true, dim=0)
        acc = float(torch.sum(y_pred == y_true)) / len(y_true)
        return y_true.cpu().numpy(), y_prob.cpu().numpy(), acc

    def mode(self, model, test=True):
        """TODO: combine this function with Trainer ?? """
        if test:
            model.eval()
        else:
            model.train()
        self.eval_history.clear()

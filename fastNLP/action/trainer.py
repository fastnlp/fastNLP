from collections import namedtuple

import numpy as np
import torch

from fastNLP.action.action import Action
from fastNLP.action.tester import Tester


class BaseTrainer(Action):
    """Base trainer for all trainers.
        Trainer receives a model and data, and then performs training.

        Subclasses must implement the following abstract methods:
        - prepare_input
        - mode
        - define_optimizer
        - data_forward
        - grad_backward
        - get_loss
    """
    TrainConfig = namedtuple("config", ["epochs", "validate", "save_when_better",
                                        "log_per_step", "log_validation", "batch_size"])

    def __init__(self, train_args):
        """
        training parameters
        """
        super(BaseTrainer, self).__init__()
        self.n_epochs = train_args.epochs
        self.validate = train_args.validate
        self.batch_size = train_args.batch_size
        self.model = None

    def train(self, network, train_data, dev_data=None):
        """General training loop.
        :param network: a model
        :param train_data: raw data for training
        :param dev_data: raw data for validation

        The method is framework independent.
        Work by calling the following methods:
            - prepare_input
            - mode
            - define_optimizer
            - data_forward
            - get_loss
            - grad_backward
            - update
        Subclasses must implement these methods with a specific framework.
        """
        self.model = network
        train_x, train_y = self.prepare_input(train_data)

        iterations, train_batch_generator = self.batchify(self.batch_size, train_x, train_y)

        test_args = Tester.TestConfig(save_output=True, validate_in_training=True,
                                      save_dev_input=True, save_loss=True, batch_size=self.batch_size)
        evaluator = Tester(test_args)

        best_loss = 1e10

        for epoch in range(self.n_epochs):
            self.mode(test=False)  # turn on the train mode

            self.define_optimizer()
            for step in range(iterations):
                batch_x, batch_y = train_batch_generator.__next__()

                prediction = self.data_forward(network, batch_x)

                loss = self.get_loss(prediction, batch_y)
                self.grad_backward(loss)
                self.update()

            if self.validate:
                if dev_data is None:
                    raise RuntimeError("No validation data provided.")
                evaluator.test(network, dev_data)
                if evaluator.loss < best_loss:
                    best_loss = evaluator.loss

        # finish training

    def prepare_input(self, data):
        """
        Perform data transformation from raw input to vector/matrix inputs.
        :param data: raw inputs
        :return (X, Y): tuple, input features and labels
        """
        raise NotImplementedError

    def mode(self, test=False):
        """
        Tell the network to be trained or not.
        :param test: bool
        """
        raise NotImplementedError

    def define_optimizer(self):
        """
        Define framework-specific optimizer specified by the models.
        """
        raise NotImplementedError

    def update(self):
        """
        Perform weight update on a model.

        For PyTorch, just call optimizer to update.
        """
        raise NotImplementedError

    def data_forward(self, network, x):
        """
        Forward pass of the data.
        :param network: a model
        :param x: input feature matrix and label vector
        :return: output by the models

        For PyTorch, just do "network(*x)"
        """
        raise NotImplementedError

    def grad_backward(self, loss):
        """
        Compute gradient with link rules.
        :param loss: a scalar where back-prop starts

        For PyTorch, just do "loss.backward()"
        """
        raise NotImplementedError

    def get_loss(self, predict, truth):
        """
        Compute loss given prediction and ground truth.
        :param predict: prediction label vector
        :param truth: ground truth label vector
        :return: a scalar
        """
        raise NotImplementedError


class ToyTrainer(BaseTrainer):
    """A simple trainer for a PyTorch model."""

    def __init__(self, train_args):
        super(ToyTrainer, self).__init__(train_args)
        self.test_mode = False
        self.weight = np.random.rand(5, 1)
        self.bias = np.random.rand()
        self._loss = 0
        self._optimizer = None

    def prepare_input(self, data):
        return data[:, :-1], data[:, -1]

    def mode(self, test=False):
        self.model.mode(test)

    def data_forward(self, network, x):
        return np.matmul(x, self.weight) + self.bias

    def grad_backward(self, loss):
        loss.backward()

    def get_loss(self, pred, truth):
        self._loss = np.mean(np.square(pred - truth))
        return self._loss

    def define_optimizer(self):
        self._optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def update(self):
        self._optimizer.step()


class WordSegTrainer(BaseTrainer):
    """
        reserve for changes
    """

    def __init__(self, train_args):
        super(WordSegTrainer, self).__init__(train_args)
        self.id2word = None
        self.word2id = None
        self.id2tag = None
        self.tag2id = None

        self.lstm_batch_size = 8
        self.lstm_seq_len = 32  # Trainer batch_size == lstm_batch_size * lstm_seq_len
        self.hidden_dim = 100
        self.lstm_num_layers = 2
        self.vocab_size = 100
        self.word_emb_dim = 100

        self.hidden = (self.to_var(torch.zeros(2, self.lstm_batch_size, self.word_emb_dim)),
                       self.to_var(torch.zeros(2, self.lstm_batch_size, self.word_emb_dim)))

        self.optimizer = None
        self._loss = None

        self.USE_GPU = False

    def to_var(self, x):
        if torch.cuda.is_available() and self.USE_GPU:
            x = x.cuda()
        return torch.autograd.Variable(x)

    def prepare_input(self, data):
        """
            perform word indices lookup to convert strings into indices
            :param data: list of string, each string contains word + space + [B, M, E, S]
            :return
        """
        word_list = []
        tag_list = []
        for line in data:
            if len(line) > 2:
                tokens = line.split("#")
                word_list.append(tokens[0])
                tag_list.append(tokens[2][0])
        self.id2word = list(set(word_list))
        self.word2id = {word: idx for idx, word in enumerate(self.id2word)}
        self.id2tag = list(set(tag_list))
        self.tag2id = {tag: idx for idx, tag in enumerate(self.id2tag)}
        words = np.array([self.word2id[w] for w in word_list]).reshape(-1, 1)
        tags = np.array([self.tag2id[t] for t in tag_list]).reshape(-1, 1)
        return words, tags

    def mode(self, test=False):
        if test:
            self.model.eval()
        else:
            self.model.train()

    def data_forward(self, network, x):
        """
        :param network: a PyTorch model
        :param x: sequence of length [batch_size], word indices
        :return:
        """
        x = x.reshape(self.lstm_batch_size, self.lstm_seq_len)
        output, self.hidden = network(x, self.hidden)
        return output

    def define_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.85)

    def get_loss(self, predict, truth):
        self._loss = torch.nn.CrossEntropyLoss(predict, truth)
        return self._loss

    def grad_backward(self, network):
        self.model.zero_grad()
        self._loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 5, norm_type=2)

    def update(self):
        self.optimizer.step()


if __name__ == "__name__":
    Config = namedtuple("config", ["epochs", "validate", "save_when_better", "log_per_step",
                                   "log_validation", "batch_size"])
    train_config = Config(epochs=5, validate=True, save_when_better=True, log_per_step=10, log_validation=True,
                          batch_size=32)
    trainer = ToyTrainer(train_config)

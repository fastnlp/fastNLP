import _pickle
import os
from datetime import timedelta
from time import time

import numpy as np
import torch
import torch.nn as nn

from fastNLP.core.action import Action
from fastNLP.core.action import RandomSampler, Batchifier
from fastNLP.core.tester import POSTester
from fastNLP.modules import utils
from fastNLP.saver.model_saver import ModelSaver


class BaseTrainer(Action):
    """Base trainer for all trainers.
        Trainer receives a model and data, and then performs training.

        Subclasses must implement the following abstract methods:
        - define_optimizer
        - grad_backward
        - get_loss
    """

    def __init__(self, train_args, action=None):
        """
        :param train_args: dict of (key, value), or dict-like object. key is str.
        :param action: (optional) an Action object that wrap most operations shared by Trainer, Tester, and Inference.

        The base trainer requires the following keys:
        - epochs: int, the number of epochs in training
        - validate: bool, whether or not to validate on dev set
        - batch_size: int
        - pickle_path: str, the path to pickle files for pre-processing
        """
        super(BaseTrainer, self).__init__()
        self.action = action if action is not None else Action()
        self.n_epochs = train_args["epochs"]
        self.batch_size = train_args["batch_size"]
        self.pickle_path = train_args["pickle_path"]

        self.validate = train_args["validate"]
        self.save_best_dev = train_args["save_best_dev"]
        self.model_saved_path = train_args["model_saved_path"]
        self.use_cuda = train_args["use_cuda"]

        self.model = None
        self.iterator = None
        self.loss_func = None
        self.optimizer = None

    def train(self, network):
        """General Training Steps
        :param network: a model

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
        # prepare model and data, transfer model to gpu if available
        if torch.cuda.is_available() and self.use_cuda:
            self.model = network.cuda()
        else:
            self.model = network

        data_train = self.prepare_input(self.pickle_path)

        # define tester over dev data
        # TODO: more flexible
        default_valid_args = {"save_output": True, "validate_in_training": True, "save_dev_input": True,
                      "save_loss": True, "batch_size": self.batch_size, "pickle_path": self.pickle_path,
                      "use_cuda": self.use_cuda}
        validator = POSTester(default_valid_args, self.action)

        # main training epochs
        iterations = len(data_train) // self.batch_size
        self.define_optimizer()

        for epoch in range(1, self.n_epochs + 1):

            # turn on network training mode; define optimizer; prepare batch iterator
            self.action.mode(self.model, test=False)
            iterator = iter(Batchifier(RandomSampler(data_train), self.batch_size, drop_last=True))

            # training iterations in one epoch
            for step in range(iterations):
                batch_x, batch_y = self.action.make_batch(iterator, data_train)

                prediction = self.data_forward(network, batch_x)

                loss = self.get_loss(prediction, batch_y)
                self.grad_backward(loss)
                self.update()

                if step % 10 == 0:
                    print("[epoch {} step {}] train loss={:.2f}".format(epoch, step, loss.data))

            if self.validate:
                validator.test(network)

                if self.save_best_dev and self.best_eval_result(validator):
                    self.save_model(network)
                    print("saved better model selected by dev")

                print("[epoch {}]".format(epoch), end=" ")
                print(validator.show_matrices())

        # finish training

    def prepare_input(self, pickle_path):
        """
        This is reserved for task-specific processing.
        :param data_path:
        :return:
        """
        return _pickle.load(open(pickle_path + "/data_train.pkl", "rb"))

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
        if self.loss_func is None:
            if hasattr(self.model, "loss"):
                self.loss_func = self.model.loss
            else:
                self.define_loss()
        return self.loss_func(predict, truth)

    def define_loss(self):
        """
            Assign an instance of loss function to self.loss_func
            E.g. self.loss_func = nn.CrossEntropyLoss()
        """
        raise NotImplementedError

    def best_eval_result(self, validator):
        """
        :param validator: a Tester instance
        :return: bool, True means current results on dev set is the best.
        """
        raise NotImplementedError

    def save_model(self, network):
        """
        :param network: the PyTorch model
        model_best_dev.pkl may be overwritten by a better model in future epochs.
        """
        ModelSaver(self.model_saved_path + "model_best_dev.pkl").save_pytorch(network)


class ToyTrainer(BaseTrainer):
    """
        An example to show the definition of Trainer.
    """

    def __init__(self, training_args):
        super(ToyTrainer, self).__init__(training_args)

    def prepare_input(self, data_path):
        data_train = _pickle.load(open(data_path + "/data_train.pkl", "rb"))
        data_dev = _pickle.load(open(data_path + "/data_train.pkl", "rb"))
        return data_train, data_dev, 0, 1

    def mode(self, test=False):
        if test:
            self.model.eval()
        else:
            self.model.train()

    def data_forward(self, network, x):
        return network(x)

    def grad_backward(self, loss):
        self.model.zero_grad()
        loss.backward()

    def get_loss(self, pred, truth):
        return np.mean(np.square(pred - truth))

    def define_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def update(self):
        self.optimizer.step()


class POSTrainer(BaseTrainer):
    """
    Trainer for Sequence Modeling

    """

    def __init__(self, train_args, action=None):
        super(POSTrainer, self).__init__(train_args, action)
        self.vocab_size = train_args["vocab_size"]
        self.num_classes = train_args["num_classes"]
        self.max_len = None
        self.mask = None
        self.best_accuracy = 0.0

    def define_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def grad_backward(self, loss):
        self.model.zero_grad()
        loss.backward()

    def update(self):
        self.optimizer.step()

    def data_forward(self, network, inputs):
        if not isinstance(inputs, tuple):
            raise RuntimeError("[fastnlp] output_length must be true for sequence modeling.")
        # unpack the returned value from make_batch
        x, seq_len = inputs[0], inputs[1]
        batch_size, max_len = x.size(0), x.size(1)
        mask = utils.seq_mask(seq_len, max_len)
        mask = mask.byte().view(batch_size, max_len)

        x = torch.Tensor(x).long()
        if torch.cuda.is_available() and self.use_cuda:
            x = x.cuda()
            mask = mask.cuda()
        self.mask = mask

        y = network(x)
        return y

    def get_loss(self, predict, truth):
        """
        Compute loss given prediction and ground truth.
        :param predict: prediction label vector, [batch_size, max_len, tag_size]
        :param truth: ground truth label vector, [batch_size, max_len]
        :return: a scalar
        """
        truth = torch.Tensor(truth)
        if torch.cuda.is_available() and self.use_cuda:
            truth = truth.cuda()
        batch_size, max_len = predict.size(0), predict.size(1)
        assert truth.shape == (batch_size, max_len)

        loss = self.model.loss(predict, truth, self.mask)
        return loss

    def best_eval_result(self, validator):
        loss, accuracy = validator.metrics()
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            return True
        else:
            return False


class LanguageModelTrainer(BaseTrainer):
    """
    Trainer for Language Model
    """

    def __init__(self, train_args):
        super(LanguageModelTrainer, self).__init__(train_args)

    def prepare_input(self, data_path):
        pass


class ClassTrainer(BaseTrainer):
    """Trainer for classification."""

    def __init__(self, train_args):
        # super(ClassTrainer, self).__init__(train_args)
        self.n_epochs = train_args["epochs"]
        self.batch_size = train_args["batch_size"]
        self.pickle_path = train_args["pickle_path"]

        if "validate" in train_args:
            self.validate = train_args["validate"]
        else:
            self.validate = False
        if "learn_rate" in train_args:
            self.learn_rate = train_args["learn_rate"]
        else:
            self.learn_rate = 1e-3
        if "momentum" in train_args:
            self.momentum = train_args["momentum"]
        else:
            self.momentum = 0.9
        if "use_cuda" in train_args:
            self.use_cuda = train_args["use_cuda"]
        else:
            self.use_cuda = True

        self.model = None
        self.iterator = None
        self.loss_func = None
        self.optimizer = None

    def train(self, network):
        """General Training Steps
        :param network: a model

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
        # prepare model and data, transfer model to gpu if available
        if torch.cuda.is_available() and self.use_cuda:
            self.model = network.cuda()
        else:
            self.model = network
        data_train, data_dev, data_test, embedding = self.prepare_input(
            self.pickle_path)

        # define tester over dev data
        # valid_args = {
        #     "save_output": True, "validate_in_training": True,
        #     "save_dev_input": True, "save_loss": True,
        #     "batch_size": self.batch_size, "pickle_path": self.pickle_path}
        # validator = POSTester(valid_args)

        # urn on network training mode, define loss and optimizer
        self.define_loss()
        self.define_optimizer()
        self.mode(test=False)

        # main training epochs
        start = time()
        n_samples = len(data_train)
        n_batches = n_samples // self.batch_size
        n_print = n_batches // 10
        for epoch in range(self.n_epochs):
            # prepare batch iterator
            self.iterator = iter(Batchifier(
                RandomSampler(data_train), self.batch_size, drop_last=False))

            # training iterations in one epoch
            step = 0
            for batch_x, batch_y in self.make_batch(data_train):
                prediction = self.data_forward(network, batch_x)

                loss = self.get_loss(prediction, batch_y)
                self.grad_backward(loss)
                self.update()

                if step % n_print == 0:
                    acc = self.get_acc(prediction, batch_y)
                    end = time()
                    diff = timedelta(seconds=round(end - start))
                    print("epoch: {:>3} step: {:>4} loss: {:>4.2}"
                          " train acc: {:>5.1%} time: {}".format(
                              epoch, step, loss, acc, diff))

                step += 1

            # if self.validate:
            #     if data_dev is None:
            #         raise RuntimeError("No validation data provided.")
            #     validator.test(network)
            #     print("[epoch {}]".format(epoch), end=" ")
            #     print(validator.show_matrices())

        # finish training

    def prepare_input(self, data_path):

        names = [
            "data_train.pkl", "data_dev.pkl",
            "data_test.pkl", "embedding.pkl"]

        files = []
        for name in names:
            file_path = os.path.join(data_path, name)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = _pickle.load(f)
            else:
                data = []
            files.append(data)

        return tuple(files)

    def mode(self, test=False):
        """
        Tell the network to be trained or not.
        :param test: bool
        """
        if test:
            self.model.eval()
        else:
            self.model.train()

    def define_loss(self):
        """
            Assign an instance of loss function to self.loss_func
            E.g. self.loss_func = nn.CrossEntropyLoss()
        """
        if self.loss_func is None:
            if hasattr(self.model, "loss"):
                self.loss_func = self.model.loss
            else:
                self.loss_func = nn.CrossEntropyLoss()

    def define_optimizer(self):
        """
        Define framework-specific optimizer specified by the models.
        """
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learn_rate,
            momentum=self.momentum)

    def data_forward(self, network, x):
        """Forward through network."""
        logits = network(x)
        return logits

    def get_loss(self, predict, truth):
        """Calculate loss."""
        return self.loss_func(predict, truth)

    def grad_backward(self, loss):
        """Compute gradient backward."""
        self.model.zero_grad()
        loss.backward()

    def update(self):
        """Apply gradient."""
        self.optimizer.step()

    def make_batch(self, data):
        """Batch and pad data."""
        for indices in self.iterator:
            batch = [data[idx] for idx in indices]
            batch_x = [sample[0] for sample in batch]
            batch_y = [sample[1] for sample in batch]
            batch_x = self.pad(batch_x)

            batch_x = torch.tensor(batch_x, dtype=torch.long)
            batch_y = torch.tensor(batch_y, dtype=torch.long)
            if torch.cuda.is_available() and self.use_cuda:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()

            yield batch_x, batch_y

    def get_acc(self, y_logit, y_true):
        """Compute accuracy."""
        y_pred = torch.argmax(y_logit, dim=-1)
        return int(torch.sum(y_true == y_pred)) / len(y_true)


if __name__ == "__name__":
    train_args = {"epochs": 1, "validate": False, "batch_size": 3, "pickle_path": "./"}
    trainer = BaseTrainer(train_args)
    data_train = [[[1, 2, 3, 4], [0]] * 10] + [[[1, 3, 5, 2], [1]] * 10]
    trainer.make_batch(data=data_train)

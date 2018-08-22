import _pickle
import copy
import os
import time
from datetime import timedelta

import numpy as np
import torch

from fastNLP.core.action import Action
from fastNLP.core.action import RandomSampler, Batchifier
from fastNLP.core.loss import Loss
from fastNLP.core.optimizer import Optimizer
from fastNLP.core.tester import SeqLabelTester, ClassificationTester
from fastNLP.modules import utils
from fastNLP.saver.logger import create_logger
from fastNLP.saver.model_saver import ModelSaver

DEFAULT_QUEUE_SIZE = 300
logger = create_logger(__name__, "./train_test.log")


class BaseTrainer(object):
    """Operations to train a model, including data loading, SGD, and validation.

        Subclasses must implement the following abstract methods:
        - grad_backward
        - get_loss
    """

    def __init__(self, **kwargs):
        """
        :param kwargs: dict of (key, value), or dict-like object. key is str.

        The base trainer requires the following keys:
        - epochs: int, the number of epochs in training
        - validate: bool, whether or not to validate on dev set
        - batch_size: int
        - pickle_path: str, the path to pickle files for pre-processing
        """
        super(BaseTrainer, self).__init__()

        """
            "default_args" provides default value for important settings. 
            The initialization arguments "kwargs" with the same key (name) will override the default value. 
            "kwargs" must have the same type as "default_args" on corresponding keys. 
            Otherwise, error will raise.
        """
        default_args = {"epochs": 3, "batch_size": 8, "validate": True, "use_cuda": True, "pickle_path": "./save/",
                        "save_best_dev": True, "model_name": "default_model_name.pkl",
                        "loss": Loss(None),
                        "optimizer": Optimizer("Adam", lr=0.001, weight_decay=0)
                        }
        """
            "required_args" is the collection of arguments that users must pass to Trainer explicitly. 
            This is used to warn users of essential settings in the training. 
            Obviously, "required_args" is the subset of "default_args". 
            The value in "default_args" to the keys in "required_args" is simply for type check. 
        """
        # TODO: required arguments
        required_args = {}

        for req_key in required_args:
            if req_key not in kwargs:
                logger.error("Trainer lacks argument {}".format(req_key))
                raise ValueError("Trainer lacks argument {}".format(req_key))

        for key in default_args:
            if key in kwargs:
                if isinstance(kwargs[key], type(default_args[key])):
                    default_args[key] = kwargs[key]
                else:
                    msg = "Argument %s type mismatch: expected %s while get %s" % (
                        key, type(default_args[key]), type(kwargs[key]))
                    logger.error(msg)
                    raise ValueError(msg)
            else:
                # BaseTrainer doesn't care about extra arguments
                pass
        print(default_args)

        self.n_epochs = default_args["epochs"]
        self.batch_size = default_args["batch_size"]
        self.pickle_path = default_args["pickle_path"]
        self.validate = default_args["validate"]
        self.save_best_dev = default_args["save_best_dev"]
        self.use_cuda = default_args["use_cuda"]
        self.model_name = default_args["model_name"]

        self._model = None
        self._loss_func = default_args["loss"].get()  # return a pytorch loss function or None
        self._optimizer = None
        self._optimizer_proto = default_args["optimizer"]

    def train(self, network, train_data, dev_data=None):
        """General Training Steps
        :param network: a model
        :param train_data: three-level list, the training set.
        :param dev_data: three-level list, the validation data (optional)

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
            self._model = network.cuda()
        else:
            self._model = network

        # define tester over dev data
        if self.validate:
            default_valid_args = {"save_output": True, "validate_in_training": True, "save_dev_input": True,
                                  "save_loss": True, "batch_size": self.batch_size, "pickle_path": self.pickle_path,
                                  "use_cuda": self.use_cuda}
            validator = self._create_validator(default_valid_args)
            logger.info("validator defined as {}".format(str(validator)))

        self.define_optimizer()
        logger.info("optimizer defined as {}".format(str(self._optimizer)))
        self.define_loss()
        logger.info("loss function defined as {}".format(str(self._loss_func)))

        # main training epochs
        n_samples = len(train_data)
        n_batches = n_samples // self.batch_size
        n_print = 1
        start = time.time()
        logger.info("training epochs started")

        for epoch in range(1, self.n_epochs + 1):
            logger.info("training epoch {}".format(epoch))

            # turn on network training mode
            self.mode(network, test=False)
            # prepare mini-batch iterator
            data_iterator = iter(Batchifier(RandomSampler(train_data), self.batch_size, drop_last=False))
            logger.info("prepared data iterator")

            self._train_step(data_iterator, network, start=start, n_print=n_print, epoch=epoch)

            if self.validate:
                logger.info("validation started")
                validator.test(network, dev_data)

                if self.save_best_dev and self.best_eval_result(validator):
                    self.save_model(network, self.model_name)
                    print("saved better model selected by dev")
                    logger.info("saved better model selected by dev")

                valid_results = validator.show_matrices()
                print("[epoch {}] {}".format(epoch, valid_results))
                logger.info("[epoch {}] {}".format(epoch, valid_results))

    def _train_step(self, data_iterator, network, **kwargs):
        """Training process in one epoch."""
        step = 0
        for batch_x, batch_y in self.make_batch(data_iterator):

            prediction = self.data_forward(network, batch_x)

            loss = self.get_loss(prediction, batch_y)
            self.grad_backward(loss)
            self.update()

            if step % kwargs["n_print"] == 0:
                end = time.time()
                diff = timedelta(seconds=round(end - kwargs["start"]))
                print_output = "[epoch: {:>3} step: {:>4}] train loss: {:>4.2} time: {}".format(
                    kwargs["epoch"], step, loss.data, diff)
                print(print_output)
                logger.info(print_output)
            step += 1

    def cross_validate(self, network, train_data_cv, dev_data_cv):
        """Training with cross validation.

        :param network: the model
        :param train_data_cv: four-level list, of shape [num_folds, num_examples, 2, ?]
        :param dev_data_cv: four-level list, of shape [num_folds, num_examples, 2, ?]

        """
        if len(train_data_cv) != len(dev_data_cv):
            logger.error("the number of folds in train and dev data unequals {}!={}".format(len(train_data_cv),
                                                                                            len(dev_data_cv)))
            raise RuntimeError("the number of folds in train and dev data unequals")
        if self.validate is False:
            logger.warn("Cross validation requires self.validate to be True. Please turn it on. ")
            print("[warning] Cross validation requires self.validate to be True. Please turn it on. ")
            self.validate = True

        n_fold = len(train_data_cv)
        logger.info("perform {} folds cross validation.".format(n_fold))
        for i in range(n_fold):
            print("CV:", i)
            logger.info("running the {} of {} folds cross validation".format(i + 1, n_fold))
            network_copy = copy.deepcopy(network)
            self.train(network_copy, train_data_cv[i], dev_data_cv[i])

    def load_train_data(self, pickle_path):
        """
        For task-specific processing.
        :param pickle_path:
        :return data_train
        """
        file_path = os.path.join(pickle_path, "data_train.pkl")
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = _pickle.load(f)
        else:
            logger.error("cannot find training data {}. invalid input path for training data.".format(file_path))
            raise RuntimeError("cannot find training data {}".format(file_path))
        return data

    def make_batch(self, iterator):
        raise NotImplementedError

    def mode(self, network, test):
        Action.mode(network, test)

    def define_optimizer(self):
        """
        Define framework-specific optimizer specified by the models.
        """
        self._optimizer = self._optimizer_proto.construct_from_pytorch(self._model.parameters())

    def update(self):
        """
        Perform weight update on a model.

        For PyTorch, just call optimizer to update.
        """
        self._optimizer.step()

    def data_forward(self, network, x):
        raise NotImplementedError

    def grad_backward(self, loss):
        """
        Compute gradient with link rules.
        :param loss: a scalar where back-prop starts

        For PyTorch, just do "loss.backward()"
        """
        self._model.zero_grad()
        loss.backward()

    def get_loss(self, predict, truth):
        """
        Compute loss given prediction and ground truth.
        :param predict: prediction label vector
        :param truth: ground truth label vector
        :return: a scalar
        """
        return self._loss_func(predict, truth)

    def define_loss(self):
        """
        if the model defines a loss, use model's loss.
        Otherwise, Trainer must has a loss argument, use it as loss.
        These two losses cannot be defined at the same time.
        Trainer does not handle loss definition or choose default losses.
        """
        if hasattr(self._model, "loss") and self._loss_func is not None:
            raise ValueError("Both the model and Trainer define loss. Please take out your loss.")

        if hasattr(self._model, "loss"):
            self._loss_func = self._model.loss
            logger.info("The model has a loss function, use it.")
        else:
            if self._loss_func is None:
                raise ValueError("Please specify a loss function.")
            logger.info("The model didn't define loss, use Trainer's loss.")

    def best_eval_result(self, validator):
        """
        :param validator: a Tester instance
        :return: bool, True means current results on dev set is the best.
        """
        raise NotImplementedError

    def save_model(self, network, model_name):
        """
        :param network: the PyTorch model
        :param model_name: str
        model_best_dev.pkl may be overwritten by a better model in future epochs.
        """
        if model_name[-4:] != ".pkl":
            model_name += ".pkl"
        ModelSaver(self.pickle_path + model_name).save_pytorch(network)

    def _create_validator(self, valid_args):
        raise NotImplementedError


class ToyTrainer(BaseTrainer):
    """
        An example to show the definition of Trainer.
    """

    def __init__(self, training_args):
        super(ToyTrainer, self).__init__(training_args)

    def load_train_data(self, data_path):
        data_train = _pickle.load(open(data_path + "/data_train.pkl", "rb"))
        data_dev = _pickle.load(open(data_path + "/data_train.pkl", "rb"))
        return data_train, data_dev, 0, 1

    def data_forward(self, network, x):
        return network(x)

    def grad_backward(self, loss):
        self._model.zero_grad()
        loss.backward()

    def get_loss(self, pred, truth):
        return np.mean(np.square(pred - truth))


class SeqLabelTrainer(BaseTrainer):
    """
    Trainer for Sequence Modeling

    """

    def __init__(self, **kwargs):
        super(SeqLabelTrainer, self).__init__(**kwargs)
        # self.vocab_size = kwargs["vocab_size"]
        # self.num_classes = kwargs["num_classes"]
        self.max_len = None
        self.mask = None
        self.best_accuracy = 0.0

    def data_forward(self, network, inputs):
        if not isinstance(inputs, tuple):
            raise RuntimeError("output_length must be true for sequence modeling. Receive {}".format(type(inputs[0])))
        # unpack the returned value from make_batch
        x, seq_len = inputs[0], inputs[1]

        batch_size, max_len = x.size(0), x.size(1)
        mask = utils.seq_mask(seq_len, max_len)
        mask = mask.byte().view(batch_size, max_len)

        if torch.cuda.is_available() and self.use_cuda:
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
        batch_size, max_len = predict.size(0), predict.size(1)
        assert truth.shape == (batch_size, max_len)

        loss = self._model.loss(predict, truth, self.mask)
        return loss

    def best_eval_result(self, validator):
        loss, accuracy = validator.metrics()
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            return True
        else:
            return False

    def make_batch(self, iterator):
        return Action.make_batch(iterator, output_length=True, use_cuda=self.use_cuda)

    def _create_validator(self, valid_args):
        return SeqLabelTester(**valid_args)


class ClassificationTrainer(BaseTrainer):
    """Trainer for classification."""

    def __init__(self, **train_args):
        super(ClassificationTrainer, self).__init__(**train_args)

        self.iterator = None
        self.loss_func = None
        self.optimizer = None
        self.best_accuracy = 0

    def data_forward(self, network, x):
        """Forward through network."""
        logits = network(x)
        return logits

    def make_batch(self, iterator):
        return Action.make_batch(iterator, output_length=False, use_cuda=self.use_cuda)

    def get_acc(self, y_logit, y_true):
        """Compute accuracy."""
        y_pred = torch.argmax(y_logit, dim=-1)
        return int(torch.sum(y_true == y_pred)) / len(y_true)

    def best_eval_result(self, validator):
        _, _, accuracy = validator.metrics()
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            return True
        else:
            return False

    def _create_validator(self, valid_args):
        return ClassificationTester(**valid_args)

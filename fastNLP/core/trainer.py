import copy
import os
import time
from datetime import timedelta

import torch
from tensorboardX import SummaryWriter

from fastNLP.core.batch import Batch
from fastNLP.core.loss import Loss
from fastNLP.core.metrics import Evaluator
from fastNLP.core.optimizer import Optimizer
from fastNLP.core.sampler import RandomSampler
from fastNLP.core.tester import SeqLabelTester, ClassificationTester
from fastNLP.saver.logger import create_logger
from fastNLP.saver.model_saver import ModelSaver

logger = create_logger(__name__, "./train_test.log")


class Trainer(object):
    """Operations of training a model, including data loading, gradient descent, and validation.

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
        super(Trainer, self).__init__()

        """
            "default_args" provides default value for important settings. 
            The initialization arguments "kwargs" with the same key (name) will override the default value. 
            "kwargs" must have the same type as "default_args" on corresponding keys. 
            Otherwise, error will raise.
        """
        default_args = {"epochs": 1, "batch_size": 2, "validate": False, "use_cuda": False, "pickle_path": "./save/",
                        "save_best_dev": False, "model_name": "default_model_name.pkl", "print_every_step": 1,
                        "loss": Loss(None),  # used to pass type check
                        "optimizer": Optimizer("Adam", lr=0.001, weight_decay=0),
                        "evaluator": Evaluator()
                        }
        """
            "required_args" is the collection of arguments that users must pass to Trainer explicitly. 
            This is used to warn users of essential settings in the training. 
            Specially, "required_args" does not have default value, so they have nothing to do with "default_args".
        """
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
                # Trainer doesn't care about extra arguments
                pass
        print(default_args)

        self.n_epochs = default_args["epochs"]
        self.batch_size = default_args["batch_size"]
        self.pickle_path = default_args["pickle_path"]
        self.validate = default_args["validate"]
        self.save_best_dev = default_args["save_best_dev"]
        self.use_cuda = default_args["use_cuda"]
        self.model_name = default_args["model_name"]
        self.print_every_step = default_args["print_every_step"]

        self._model = None
        self._loss_func = default_args["loss"].get()  # return a pytorch loss function or None
        self._optimizer = None
        self._optimizer_proto = default_args["optimizer"]
        self._evaluator = default_args["evaluator"]
        self._summary_writer = SummaryWriter(self.pickle_path + 'tensorboard_logs')
        self._graph_summaried = False
        self._best_accuracy = 0.0

    def train(self, network, train_data, dev_data=None):
        """General Training Procedure

        :param network: a model
        :param train_data: a DataSet instance, the training data
        :param dev_data: a DataSet instance, the validation data (optional)
        """
        # transfer model to gpu if available
        if torch.cuda.is_available() and self.use_cuda:
            self._model = network.cuda()
            # self._model is used to access model-specific loss
        else:
            self._model = network

        # define Tester over dev data
        if self.validate:
            default_valid_args = {"batch_size": self.batch_size, "pickle_path": self.pickle_path,
                                  "use_cuda": self.use_cuda, "evaluator": self._evaluator}
            validator = self._create_validator(default_valid_args)
            logger.info("validator defined as {}".format(str(validator)))

        # optimizer and loss
        self.define_optimizer()
        logger.info("optimizer defined as {}".format(str(self._optimizer)))
        self.define_loss()
        logger.info("loss function defined as {}".format(str(self._loss_func)))

        # main training procedure
        start = time.time()
        logger.info("training epochs started")
        for epoch in range(1, self.n_epochs + 1):
            logger.info("training epoch {}".format(epoch))

            # turn on network training mode
            self.mode(network, is_test=False)
            # prepare mini-batch iterator
            data_iterator = Batch(train_data, batch_size=self.batch_size, sampler=RandomSampler(),
                                  use_cuda=self.use_cuda)
            logger.info("prepared data iterator")

            # one forward and backward pass
            self._train_step(data_iterator, network, start=start, n_print=self.print_every_step, epoch=epoch)

            # validation
            if self.validate:
                if dev_data is None:
                    raise RuntimeError(
                        "self.validate is True in trainer, but dev_data is None. Please provide the validation data.")
                logger.info("validation started")
                validator.test(network, dev_data)

                if self.save_best_dev and self.best_eval_result(validator):
                    self.save_model(network, self.model_name)
                    print("Saved better model selected by validation.")
                    logger.info("Saved better model selected by validation.")

                valid_results = validator.show_metrics()
                print("[epoch {}] {}".format(epoch, valid_results))
                logger.info("[epoch {}] {}".format(epoch, valid_results))

    def _train_step(self, data_iterator, network, **kwargs):
        """Training process in one epoch.

            kwargs should contain:
                - n_print: int, print training information every n steps.
                - start: time.time(), the starting time of this step.
                - epoch: int,
        """
        step = 0
        for batch_x, batch_y in data_iterator:

            prediction = self.data_forward(network, batch_x)

            loss = self.get_loss(prediction, batch_y)
            self.grad_backward(loss)
            self.update()
            self._summary_writer.add_scalar("loss", loss.item(), global_step=step)

            if kwargs["n_print"] > 0 and step % kwargs["n_print"] == 0:
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

    def mode(self, model, is_test=False):
        """Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param is_test: bool, whether in test mode or not.

        """
        if is_test:
            model.eval()
        else:
            model.train()

    def define_optimizer(self):
        """Define framework-specific optimizer specified by the models.

        """
        self._optimizer = self._optimizer_proto.construct_from_pytorch(self._model.parameters())

    def update(self):
        """Perform weight update on a model.

        For PyTorch, just call optimizer to update.
        """
        self._optimizer.step()

    def data_forward(self, network, x):
        y = network(**x)
        if not self._graph_summaried:
            # self._summary_writer.add_graph(network, x, verbose=False)
            self._graph_summaried = True
        return y

    def grad_backward(self, loss):
        """Compute gradient with link rules.

        :param loss: a scalar where back-prop starts

        For PyTorch, just do "loss.backward()"
        """
        self._model.zero_grad()
        loss.backward()

    def get_loss(self, predict, truth):
        """Compute loss given prediction and ground truth.

        :param predict: prediction label vector
        :param truth: ground truth label vector
        :return: a scalar
        """
        if len(truth) > 1:
            raise NotImplementedError("Not ready to handle multi-labels.")
        truth = list(truth.values())[0] if len(truth) > 0 else None
        return self._loss_func(predict, truth)

    def define_loss(self):
        """Define a loss for the trainer.

        If the model defines a loss, use model's loss.
        Otherwise, Trainer must has a loss argument, use it as loss.
        These two losses cannot be defined at the same time.
        Trainer does not handle loss definition or choose default losses.
        """
        # if hasattr(self._model, "loss") and self._loss_func is not None:
        #    raise ValueError("Both the model and Trainer define loss. Please take out your loss.")

        if hasattr(self._model, "loss"):
            self._loss_func = self._model.loss
            logger.info("The model has a loss function, use it.")
        else:
            if self._loss_func is None:
                raise ValueError("Please specify a loss function.")
            logger.info("The model didn't define loss, use Trainer's loss.")

    def best_eval_result(self, validator):
        """Check if the current epoch yields better validation results.

        :param validator: a Tester instance
        :return: bool, True means current results on dev set is the best.
        """
        loss, accuracy = validator.metrics
        if accuracy > self._best_accuracy:
            self._best_accuracy = accuracy
            return True
        else:
            return False

    def save_model(self, network, model_name):
        """Save this model with such a name.
        This method may be called multiple times by Trainer to overwritten a better model.

        :param network: the PyTorch model
        :param model_name: str
        """
        if model_name[-4:] != ".pkl":
            model_name += ".pkl"
        ModelSaver(os.path.join(self.pickle_path, model_name)).save_pytorch(network)

    def _create_validator(self, valid_args):
        raise NotImplementedError


class SeqLabelTrainer(Trainer):
    """Trainer for Sequence Labeling

    """
    def __init__(self, **kwargs):
        print(
            "[FastNLP Warning] SeqLabelTrainer will be deprecated. Please use Trainer directly.")
        super(SeqLabelTrainer, self).__init__(**kwargs)

    def _create_validator(self, valid_args):
        return SeqLabelTester(**valid_args)


class ClassificationTrainer(Trainer):
    """Trainer for text classification."""

    def __init__(self, **train_args):
        print(
            "[FastNLP Warning] ClassificationTrainer will be deprecated. Please use Trainer directly.")
        super(ClassificationTrainer, self).__init__(**train_args)

    def _create_validator(self, valid_args):
        return ClassificationTester(**valid_args)

import os
import time
from datetime import timedelta, datetime

import torch
from tensorboardX import SummaryWriter

from fastNLP.core.batch import Batch
from fastNLP.core.loss import Loss
from fastNLP.core.metrics import Evaluator
from fastNLP.core.optimizer import Optimizer
from fastNLP.core.sampler import BucketSampler
from fastNLP.core.tester import SeqLabelTester, ClassificationTester, SNLITester
from fastNLP.core.tester import Tester
from fastNLP.saver.logger import create_logger
from fastNLP.saver.model_saver import ModelSaver

logger = create_logger(__name__, "./train_test.log")
logger.disabled = True


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
                        "valid_step": 500, "eval_sort_key": 'acc',
                        "loss": Loss(None),  # used to pass type check
                        "optimizer": Optimizer("Adam", lr=0.001, weight_decay=0),
                        "eval_batch_size": 64,
                        "evaluator": Evaluator(),
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
        print("Training Args {}".format(default_args))
        logger.info("Training Args {}".format(default_args))

        self.n_epochs = int(default_args["epochs"])
        self.batch_size = int(default_args["batch_size"])
        self.eval_batch_size = int(default_args['eval_batch_size'])
        self.pickle_path = default_args["pickle_path"]
        self.validate = default_args["validate"]
        self.save_best_dev = default_args["save_best_dev"]
        self.use_cuda = default_args["use_cuda"]
        self.model_name = default_args["model_name"]
        self.print_every_step = int(default_args["print_every_step"])
        self.valid_step = int(default_args["valid_step"])
        if self.validate is not None:
            assert self.valid_step > 0

        self._model = None
        self._loss_func = default_args["loss"].get()  # return a pytorch loss function or None
        self._optimizer = None
        self._optimizer_proto = default_args["optimizer"]
        self._evaluator = default_args["evaluator"]
        self._summary_writer = SummaryWriter(self.pickle_path + 'tensorboard_logs')
        self._graph_summaried = False
        self._best_accuracy = 0.0
        self.eval_sort_key = default_args['eval_sort_key']
        self.validator = None
        self.epoch = 0
        self.step = 0

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

        print(self._model)

        # define Tester over dev data
        self.dev_data = None
        if self.validate:
            default_valid_args = {"batch_size": self.eval_batch_size, "pickle_path": self.pickle_path,
                                  "use_cuda": self.use_cuda, "evaluator": self._evaluator}
            if self.validator is None:
                self.validator = self._create_validator(default_valid_args)
            logger.info("validator defined as {}".format(str(self.validator)))
            self.dev_data = dev_data

        # optimizer and loss
        self.define_optimizer()
        logger.info("optimizer defined as {}".format(str(self._optimizer)))
        self.define_loss()
        logger.info("loss function defined as {}".format(str(self._loss_func)))

        # turn on network training mode
        self.mode(network, is_test=False)

        # main training procedure
        start = time.time()
        self.start_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        print("training epochs started " + self.start_time)
        logger.info("training epochs started " + self.start_time)
        self.epoch, self.step = 1, 0
        while self.epoch <= self.n_epochs:
            logger.info("training epoch {}".format(self.epoch))

            # prepare mini-batch iterator
            data_iterator = Batch(train_data, batch_size=self.batch_size,
                                  sampler=BucketSampler(10, self.batch_size, "word_seq_origin_len"),
                                  use_cuda=self.use_cuda)
            logger.info("prepared data iterator")

            # one forward and backward pass
            self._train_step(data_iterator, network, start=start, n_print=self.print_every_step, dev_data=dev_data)

            # validation
            if self.validate:
                self.valid_model()
            self.save_model(self._model, 'training_model_' + self.start_time)
            self.epoch += 1

    def _train_step(self, data_iterator, network, **kwargs):
        """Training process in one epoch.

            kwargs should contain:
                - n_print: int, print training information every n steps.
                - start: time.time(), the starting time of this step.
                - epoch: int,
        """
        for batch_x, batch_y in data_iterator:
            prediction = self.data_forward(network, batch_x)

            # TODO: refactor self.get_loss
            loss = prediction["loss"] if "loss" in prediction else self.get_loss(prediction, batch_y)
            # acc = self._evaluator([{"predict": prediction["predict"]}], [{"truth": batch_x["truth"]}])

            self.grad_backward(loss)
            self.update()
            self._summary_writer.add_scalar("loss", loss.item(), global_step=self.step)
            for name, param in self._model.named_parameters():
                if param.requires_grad:
<<<<<<< HEAD
                    # self._summary_writer.add_scalar(name + "_mean", param.mean(), global_step=step)
                    # self._summary_writer.add_scalar(name + "_std", param.std(), global_step=step)
                    # self._summary_writer.add_scalar(name + "_grad_sum", param.sum(), global_step=step)
                    pass

            if kwargs["n_print"] > 0 and step % kwargs["n_print"] == 0:
=======
                    self._summary_writer.add_scalar(name + "_mean", param.mean(), global_step=self.step)
                    # self._summary_writer.add_scalar(name + "_std", param.std(), global_step=self.step)
                    # self._summary_writer.add_scalar(name + "_grad_sum", param.sum(), global_step=self.step)
            if kwargs["n_print"] > 0 and self.step % kwargs["n_print"] == 0:
>>>>>>> 5924fe0... fix and update tester, trainer, seq_model, add parser pipeline builder
                end = time.time()
                diff = timedelta(seconds=round(end - kwargs["start"]))
                print_output = "[epoch: {:>3} step: {:>4}] train loss: {:>4.6} time: {}".format(
                    self.epoch, self.step, loss.data, diff)
                print(print_output)
                logger.info(print_output)
            if self.validate and self.valid_step > 0 and self.step > 0 and self.step % self.valid_step == 0:
                self.valid_model()
            self.step += 1

    def valid_model(self):
        if self.dev_data is None:
            raise RuntimeError(
                "self.validate is True in trainer, but dev_data is None. Please provide the validation data.")
        logger.info("validation started")
        res = self.validator.test(self._model, self.dev_data)
        for name, num in res.items():
            self._summary_writer.add_scalar("valid_{}".format(name), num, global_step=self.step)
        if self.save_best_dev and self.best_eval_result(res):
            logger.info('save best result! {}'.format(res))
            print('save best result! {}'.format(res))
            self.save_model(self._model, 'best_model_' + self.start_time)
        return res

    def mode(self, model, is_test=False):
        """Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param is_test: bool, whether in test mode or not.

        """
        if is_test:
            model.eval()
        else:
            model.train()

    def define_optimizer(self, optim=None):
        """Define framework-specific optimizer specified by the models.

        """
        if optim is not None:
            # optimizer constructed by user
            self._optimizer = optim
        elif self._optimizer is None:
            # optimizer constructed by proto
            self._optimizer = self._optimizer_proto.construct_from_pytorch(self._model.parameters())
        return self._optimizer

    def update(self):
        """Perform weight update on a model.

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
        if isinstance(predict, dict) and isinstance(truth, dict):
            return self._loss_func(**predict, **truth)
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

    def best_eval_result(self, metrics):
        """Check if the current epoch yields better validation results.

        :param validator: a Tester instance
        :return: bool, True means current results on dev set is the best.
        """
        if isinstance(metrics, tuple):
            loss, metrics = metrics

        if isinstance(metrics, dict):
            if len(metrics) == 1:
                accuracy = list(metrics.values())[0]
            else:
                accuracy = metrics[self.eval_sort_key]
        else:
            accuracy = metrics

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
        return Tester(**valid_args)

    def set_validator(self, validor):
        self.validator = validor


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


class SNLITrainer(Trainer):
    """Trainer for text SNLI."""

    def __init__(self, **train_args):
        print(
            "[FastNLP Warning] SNLITrainer will be deprecated. Please use Trainer directly.")
        super(SNLITrainer, self).__init__(**train_args)

    def _create_validator(self, valid_args):
        return SNLITester(**valid_args)

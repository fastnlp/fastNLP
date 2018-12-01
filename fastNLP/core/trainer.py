import os
import time
import warnings
from datetime import datetime
from datetime import timedelta

import torch
from tensorboardX import SummaryWriter
from torch import nn

from fastNLP.core.batch import Batch
from fastNLP.core.dataset import DataSet
from fastNLP.core.losses import _prepare_losser
from fastNLP.core.metrics import _prepare_metrics
from fastNLP.core.optimizer import Adam
from fastNLP.core.sampler import RandomSampler
from fastNLP.core.sampler import SequentialSampler
from fastNLP.core.tester import Tester
from fastNLP.core.utils import CheckError
from fastNLP.core.utils import _build_args
from fastNLP.core.utils import _check_arg_dict_list
from fastNLP.core.utils import _move_dict_value_to_device
from fastNLP.core.utils import get_func_signature
from fastNLP.core.dataset import DataSet
from fastNLP.core.losses import _prepare_losser
from fastNLP.core.metrics import _prepare_metrics
from fastNLP.core.utils import CheckError
from fastNLP.core.utils import _check_loss_evaluate
from fastNLP.core.utils import _check_forward_error

class Trainer(object):
    """Main Training Loop

    """

    def __init__(self, train_data, model, losser=None, metrics=None, n_epochs=3, batch_size=32, print_every=-1,
                 validate_every=-1,
                 dev_data=None, use_cuda=False, save_path="./save",
                 optimizer=Adam(lr=0.01, weight_decay=0), check_code_level=0,
                 metric_key=None,
                 **kwargs):
        super(Trainer, self).__init__()

        if not isinstance(train_data, DataSet):
            raise TypeError(f"The type of train_data must be fastNLP.DataSet, got {type(train_data)}.")
        if not isinstance(model, nn.Module):
            raise TypeError(f"The type of model must be torch.nn.Module, got {type(model)}.")

        # check metrics and dev_data
        if (not metrics) and dev_data is not None:
            raise ValueError("No metric for dev_data evaluation.")
        if metrics and (dev_data is None):
            raise ValueError("No dev_data for evaluations, pass dev_data or set metrics to None. ")

        # prepare evaluate
        metrics = _prepare_metrics(metrics)

        # parse metric_key
        # increase_better is True. It means the exp result gets better if the indicator increases.
        # It is true by default.
        self.increase_better = False if metric_key[0] == "-" else True
        self.metric_key = metric_key[1:] if metric_key[0] == "+" or metric_key[0] == "-" else metric_key

        # prepare loss
        losser = _prepare_losser(losser)

        if check_code_level>-1:
            _check_code(dataset=train_data, model=model, losser=losser, metrics=metrics, dev_data=dev_data,
                        check_level=check_code_level)

        self.train_data = train_data
        self.dev_data = dev_data  # If None, No validation.
        self.model = model
        self.losser = losser
        self.metrics = metrics
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.use_cuda = bool(use_cuda)
        self.save_path = save_path
        self.print_every = int(print_every)
        self.validate_every = int(validate_every)
        self.best_metric_indicator = None

        self._model_device = model.parameters().__next__().device

        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer.construct_from_pytorch(self.model.parameters())

        if self.dev_data is not None:
            self.tester = Tester(model=self.model,
                                 data=self.dev_data,
                                 metrics=self.metrics,
                                 batch_size=self.batch_size,
                                 use_cuda=self.use_cuda)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.step = 0
        self.start_time = None  # start timestamp

        # print(self.__dict__)

    def train(self):
        """Start Training.

        :return:
        """
        try:
            if torch.cuda.is_available() and self.use_cuda:
                self.model = self.model.cuda()

            self._mode(self.model, is_test=False)

            start = time.time()
            self.start_time = str(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            print("training epochs started " + self.start_time)
            if self.save_path is None:
                class psudoSW:
                    def __getattr__(self, item):
                        def pass_func(*args, **kwargs):
                            pass

                        return pass_func

                self._summary_writer = psudoSW()
            else:
                path = os.path.join(self.save_path, 'tensorboard_logs_{}'.format(self.start_time))
                self._summary_writer = SummaryWriter(path)

            epoch = 1
            while epoch <= self.n_epochs:

                data_iterator = Batch(self.train_data, batch_size=self.batch_size, sampler=RandomSampler(),
                                      as_numpy=False)

                self._train_epoch(data_iterator, self.model, epoch, start)

                # validate_every override validation at end of epochs
                if self.dev_data and self.validate_every <= 0:
                    self._do_validation()
                epoch += 1
        finally:
            self._summary_writer.close()
            del self._summary_writer

    def _train_epoch(self, data_iterator, model, epoch, start):
        """Training process in one epoch.

            kwargs should contain:
                - n_print: int, print training information every n steps.
                - start: time.time(), the starting time of this step.
                - epoch: int,
        """
        for batch_x, batch_y in data_iterator:
            # TODO 这里可能会遇到问题，万一用户在model内部修改了prediction的device就会有问题
            _move_dict_value_to_device(self._model_device, batch_x, batch_y)
            prediction = self._data_forward(model, batch_x)
            loss = self._compute_loss(prediction, batch_y)
            self._grad_backward(loss)
            self._update()
            self._summary_writer.add_scalar("loss", loss.item(), global_step=self.step)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self._summary_writer.add_scalar(name + "_mean", param.mean(), global_step=self.step)
                    # self._summary_writer.add_scalar(name + "_std", param.std(), global_step=self.step)
                    # self._summary_writer.add_scalar(name + "_grad_sum", param.sum(), global_step=self.step)
            if self.print_every > 0 and self.step % self.print_every == 0:
                end = time.time()
                diff = timedelta(seconds=round(end - start))
                print_output = "[epoch: {:>3} step: {:>4}] train loss: {:>4.6} time:  {}".format(
                    epoch, self.step, loss.data, diff)
                print(print_output)

            if self.validate_every > 0 and self.step % self.validate_every == 0:
                self._do_validation()

            self.step += 1

    def _do_validation(self):
        res = self.tester.test()
        for name, num in res.items():
            self._summary_writer.add_scalar("valid_{}".format(name), num, global_step=self.step)
        if self.save_path is not None and self._better_eval_result(res):
            self._save_model(self.model,
                            "best_" + "_".join([self.model.__class__.__name__, self.metric_key, self.start_time]))

    def _mode(self, model, is_test=False):
        """Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param is_test: bool, whether in test mode or not.

        """
        if is_test:
            model.eval()
        else:
            model.train()

    def _update(self):
        """Perform weight update on a model.

        """
        self.optimizer.step()

    def _data_forward(self, network, x):
        x = _build_args(network.forward, **x)
        y = network(**x)
        if not isinstance(y, dict):
            raise TypeError(f"The return value of {get_func_signature(network.forward)} should be dict, got {type(y)}.")
        return y

    def _grad_backward(self, loss):
        """Compute gradient with link rules.

        :param loss: a scalar where back-prop starts

        For PyTorch, just do "loss.backward()"
        """
        self.model.zero_grad()
        loss.backward()

    def _compute_loss(self, predict, truth):
        """Compute loss given prediction and ground truth.

        :param predict: prediction dict, produced by model.forward
        :param truth: ground truth dict, produced by batch_y
        :return: a scalar
        """
        return self.losser(predict, truth)

    def _save_model(self, model, model_name, only_param=False):
        model_name = os.path.join(self.save_path, model_name)
        if only_param:
            torch.save(model.state_dict(), model_name)
        else:
            torch.save(model, model_name)

    def _better_eval_result(self, metrics):
        """Check if the current epoch yields better validation results.

        :return bool value: True means current results on dev set is the best.
        """
        if isinstance(metrics, tuple):
            loss, metrics = metrics

        if isinstance(metrics, dict):
            if len(metrics) == 1:
                # only single metric, just use it
                metric_dict = list(metrics.values())[0]
                metrics_name = list(metrics.keys())[0]
            else:
                metrics_name = self.metrics[0].__class__.__name__
                if metrics_name not in metrics:
                    raise RuntimeError(f"{metrics_name} is chosen to do validation, but got {metrics}")
                metric_dict = metrics[metrics_name]

            if len(metric_dict) == 1:
                indicator_val, indicator = list(metric_dict.values())[0], list(metric_dict.keys())[0]
            elif len(metric_dict) > 1 and self.metric_key is None:
                raise RuntimeError(
                    f"Got multiple metric keys: {metric_dict}, but metric_key is not set. Which one to use?")
            else:
                # metric_key is set
                if self.metric_key not in metric_dict:
                    raise RuntimeError(f"matric key {self.metric_key} not found in {metric_dict}")
                indicator_val = metric_dict[self.metric_key]

            is_better = True
            if self.best_metric_indicator is None:
                # first-time validation
                self.best_metric_indicator = indicator_val
            else:
                if self.increase_better is True:
                    if indicator_val > self.best_metric_indicator:
                        self.best_metric_indicator = indicator_val
                    else:
                        is_better = False
                else:
                    if indicator_val < self.best_metric_indicator:
                        self.best_metric_indicator = indicator_val
                    else:
                        is_better = False
            return is_better


DEFAULT_CHECK_BATCH_SIZE = 2
DEFAULT_CHECK_NUM_BATCH = 2

def _check_code(dataset, model, losser, metrics, batch_size=DEFAULT_CHECK_BATCH_SIZE,
                dev_data=None,
                check_level=0):
    # check get_loss 方法
    model_devcie = model.parameters().__next__().device

    batch = Batch(dataset=dataset, batch_size=batch_size, sampler=SequentialSampler())
    for batch_count, (batch_x, batch_y) in enumerate(batch):
        _move_dict_value_to_device(model_devcie, batch_x, batch_y)
        # forward check
        if batch_count==0:
            _check_forward_error(forward_func=model.forward, check_level=check_level,
                                 batch_x=batch_x)

        refined_batch_x = _build_args(model.forward, **batch_x)
        output = model(**refined_batch_x)
        func_signature = get_func_signature(model.forward)
        if not isinstance(output, dict):
            raise TypeError(f"The return value of {func_signature} should be `dict`, not `{type(output)}`.")

        # loss check
        try:
            loss = losser(output, batch_y)
            # check loss output
            if batch_count == 0:
                if not isinstance(loss, torch.Tensor):
                    raise TypeError(
                        f"The return value of {get_func_signature(losser.get_loss)} should be `torch.Tensor`, "
                        f"but got `{type(loss)}`.")
                if len(loss.size()) != 0:
                    raise ValueError(
                        f"The size of return value of {get_func_signature(losser.get_loss)} is {loss.size()}, "
                        f"should be torch.size([])")
            loss.backward()
        except CheckError as e:
            pre_func_signature = get_func_signature(model.forward)
            _check_loss_evaluate(prev_func_signature=pre_func_signature, func_signature=e.func_signature,
                                 check_res=e.check_res, output=output, batch_y=batch_y,
                                 check_level=check_level)
        model.zero_grad()
        if batch_count + 1 >= DEFAULT_CHECK_NUM_BATCH:
            break

    if dev_data is not None:
        tester = Tester(data=dataset[:batch_size * DEFAULT_CHECK_NUM_BATCH], model=model, metrics=metrics,
                        batch_size=batch_size, verbose=-1)
        tester.test()



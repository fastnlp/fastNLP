import os
import time
from datetime import datetime
from datetime import timedelta

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from tqdm.autonotebook import tqdm

from fastNLP.core.batch import Batch
from fastNLP.core.dataset import DataSet
from fastNLP.core.losses import _prepare_losser
from fastNLP.core.metrics import _prepare_metrics
from fastNLP.core.optimizer import Adam
from fastNLP.core.sampler import BaseSampler
from fastNLP.core.sampler import RandomSampler
from fastNLP.core.sampler import SequentialSampler
from fastNLP.core.tester import Tester
from fastNLP.core.utils import CheckError
from fastNLP.core.utils import _build_args
from fastNLP.core.utils import _check_forward_error
from fastNLP.core.utils import _check_loss_evaluate
from fastNLP.core.utils import _move_dict_value_to_device
from fastNLP.core.utils import get_func_signature


class Trainer(object):
    """Main Training Loop

    """
    def __init__(self, train_data, model, loss=None, metrics=None, n_epochs=3, batch_size=32, print_every=50,
                 validate_every=-1, dev_data=None, use_cuda=False, save_path=None,
                 optimizer=Adam(lr=0.01, weight_decay=0), check_code_level=0,
                 metric_key=None, sampler=RandomSampler(), use_tqdm=True):
        """

        :param DataSet train_data: the training data
        :param torch.nn.modules.module model: a PyTorch model
        :param LossBase loss: a loss object
        :param MetricBase or List[MetricBase] metrics: a metric object or a list of metrics
        :param int n_epochs: the number of training epochs
        :param int batch_size: batch size for training and validation
        :param int print_every: step interval to print next training information. Default: -1(no print).
        :param int validate_every: step interval to do next validation. Default: -1(validate every epoch).
        :param DataSet dev_data: the validation data
        :param use_cuda:
        :param save_path: file path to save models
        :param Optimizer optimizer: an optimizer object
        :param int check_code_level: level of FastNLP code checker. -1: don't check, 0: ignore. 1: warning. 2: strict.
            `ignore` will not check unused field; `warning` when warn if some field are not used; `strict` means
            it will raise error if some field are not used.
        :param str metric_key: a single indicator used to decide the best model based on metric results. It must be one
            of the keys returned by the FIRST metric in `metrics`. If the overall result gets better if the indicator gets
            smaller, add a `-` character in front of the string. For example
                ::
                    metric_key="-PPL"   # language model gets better as perplexity gets smaller
        :param sampler: method used to generate batch data.
        :param use_tqdm: boolean, use tqdm to show train progress.

        """
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

        # check save_path
        if not (save_path is None or isinstance(save_path, str)):
            raise ValueError("save_path can only be None or `str`.")
        # prepare evaluate
        metrics = _prepare_metrics(metrics)

        # parse metric_key
        # increase_better is True. It means the exp result gets better if the indicator increases.
        # It is true by default.
        self.increase_better = True
        if metric_key is not None:
            self.increase_better = False if metric_key[0] == "-" else True
            self.metric_key = metric_key[1:] if metric_key[0] == "+" or metric_key[0] == "-" else metric_key
        elif len(metrics) > 0:
            self.metric_key = metrics[0].__class__.__name__.lower().strip('metric')

        # prepare loss
        losser = _prepare_losser(loss)

        # sampler check
        if not isinstance(sampler, BaseSampler):
            raise ValueError("The type of sampler should be fastNLP.BaseSampler, got {}.".format(type(sampler)))

        if check_code_level > -1:
            _check_code(dataset=train_data, model=model, losser=losser, metrics=metrics, dev_data=dev_data,
                        metric_key=metric_key, check_level=check_code_level,
                        batch_size=min(batch_size, DEFAULT_CHECK_BATCH_SIZE))

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
        self.sampler = sampler

        if isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer
        else:
            self.optimizer = optimizer.construct_from_pytorch(self.model.parameters())

        self.use_tqdm = use_tqdm
        if self.use_tqdm:
            tester_verbose = 0
            self.print_every = abs(self.print_every)
        else:
            tester_verbose = 1

        if self.dev_data is not None:
            self.tester = Tester(model=self.model,
                                 data=self.dev_data,
                                 metrics=self.metrics,
                                 batch_size=self.batch_size,
                                 use_cuda=self.use_cuda,
                                 verbose=tester_verbose)

        self.step = 0
        self.start_time = None  # start timestamp

    def train(self, load_best_model=True):
        """

        开始训练过程。主要有以下几个步骤
        for epoch in range(num_epochs):
            (1) 使用Batch从DataSet中按批取出数据，并自动对DataSet中dtype为float, int的fields进行padding。并转换为Tensor。非
                    float，int类型的参数将不会被转换为Tensor，且不进行padding
            for batch_x, batch_y in Batch(DataSet):
                # batch_x中为设置为input的field
                # batch_y中为设置为target的field
                (2) 将batch_x的数据送入到model.forward函数中，并获取结果
                (3) 将batch_y与model.forward的结果一并送入loss中计算loss
                (4) 获取到loss之后，进行反向求导并更新梯度
        if dev_data is not None:
            根据metrics进行evaluation，并根据是否提供了save_path判断是否存储模型

        :param load_best_model: 该参数只有在初始化提供了dev_data的情况下有效，如果True, trainer将在返回之前重新加载dev表现最好的
            模型参数。

        将会返回一个字典类型的数据, 内含以下内容：
            seconds: float, 表示训练时长
            以下三个内容只有在提供了dev_data的情况下会有。
            best_eval: Dict of Dict, 表示evaluation的结果
            best_epoch: int，在第几个epoch取得的最佳值
            best_step: int, 在第几个step(batch)更新取得的最佳值

        return dict:
        """
        results = {}
        try:
            if torch.cuda.is_available() and self.use_cuda:
                self.model = self.model.cuda()
            self._model_device = self.model.parameters().__next__().device

            self._mode(self.model, is_test=False)

            self.start_time = str(datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
            start_time = time.time()
            print("training epochs started " + self.start_time, flush=True)
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
            if self.use_tqdm:
                self._tqdm_train()
            else:
                self._print_train()
            if self.dev_data is not None:
                print("\nIn Epoch:{}/Step:{}, got best dev performance:".format(self.best_dev_epoch, self.best_dev_step) +
                      self.tester._format_eval_results(self.best_dev_perf),)
                results['best_eval'] = self.best_dev_perf
                results['best_epoch'] = self.best_dev_epoch
                results['best_step'] = self.best_dev_step
                if load_best_model:
                    model_name = "best_" + "_".join([self.model.__class__.__name__, self.metric_key, self.start_time])
                    # self._load_model(self.model, model_name)
                    print("Reloaded the best model.")
        finally:
            self._summary_writer.close()
            del self._summary_writer
        results['seconds'] = round(time.time() - start_time, 2)

        return results

    def _tqdm_train(self):
        self.step = 0
        data_iterator = Batch(self.train_data, batch_size=self.batch_size, sampler=self.sampler,
                              as_numpy=False)
        total_steps = data_iterator.num_batches*self.n_epochs
        with tqdm(total=total_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True) as pbar:
            avg_loss = 0
            for epoch in range(1, self.n_epochs+1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.n_epochs))
                for batch_x, batch_y in data_iterator:
                    _move_dict_value_to_device(batch_x, batch_y, device=self._model_device)
                    prediction = self._data_forward(self.model, batch_x)
                    loss = self._compute_loss(prediction, batch_y)
                    avg_loss += loss.item()
                    self._grad_backward(loss)
                    self._update()
                    self._summary_writer.add_scalar("loss", loss.item(), global_step=self.step)
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            self._summary_writer.add_scalar(name + "_mean", param.mean(), global_step=self.step)
                            # self._summary_writer.add_scalar(name + "_std", param.std(), global_step=self.step)
                            # self._summary_writer.add_scalar(name + "_grad_sum", param.sum(), global_step=self.step)
                    if (self.step+1) % self.print_every == 0:
                        pbar.set_postfix_str("loss:{0:<6.5f}".format(avg_loss / self.print_every))
                        avg_loss = 0
                        pbar.update(self.print_every)
                    self.step += 1
                    if self.validate_every > 0 and self.step % self.validate_every == 0 \
                            and self.dev_data is not None:
                        eval_res = self._do_validation(epoch=epoch, step=self.step)
                        eval_str = "Epoch {}/{}. Step:{}/{}. ".format(epoch, self.n_epochs, self.step, total_steps) + \
                                   self.tester._format_eval_results(eval_res)
                        pbar.write(eval_str)
                if self.validate_every < 0 and self.dev_data:
                    eval_res = self._do_validation(epoch=epoch, step=self.step)
                    eval_str = "Epoch {}/{}. Step:{}/{}. ".format(epoch, self.n_epochs, self.step, total_steps) + \
                               self.tester._format_eval_results(eval_res)
                    pbar.write(eval_str)
                if epoch!=self.n_epochs:
                    data_iterator = Batch(self.train_data, batch_size=self.batch_size, sampler=self.sampler,
                                          as_numpy=False)
            pbar.close()

    def _print_train(self):
        epoch = 1
        start = time.time()
        while epoch <= self.n_epochs:

            data_iterator = Batch(self.train_data, batch_size=self.batch_size, sampler=self.sampler,
                                  as_numpy=False)

            for batch_x, batch_y in data_iterator:
                # TODO 这里可能会遇到问题，万一用户在model内部修改了prediction的device就会有问题
                _move_dict_value_to_device(batch_x, batch_y, device=self._model_device)
                prediction = self._data_forward(self.model, batch_x)
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

                if (self.validate_every > 0 and self.step % self.validate_every == 0 and
                        self.dev_data is not None):
                    self._do_validation(epoch=epoch, step=self.step)

                self.step += 1

            # validate_every override validation at end of epochs
            if self.dev_data and self.validate_every <= 0:
                self._do_validation(epoch=epoch, step=self.step)
            epoch += 1

    def _do_validation(self, epoch, step):
        res = self.tester.test()
        for name, metric in res.items():
            for metric_key, metric_val in metric.items():
                self._summary_writer.add_scalar("valid_{}_{}".format(name, metric_key), metric_val,
                                                global_step=self.step)
        if self._better_eval_result(res):
            if self.save_path is not None:
                self._save_model(self.model,
                             "best_" + "_".join([self.model.__class__.__name__, self.metric_key, self.start_time]))

            self.best_dev_perf = res
            self.best_dev_epoch = epoch
            self.best_dev_step = step
        return res

    def _mode(self, model, is_test=False):
        """Train mode or Test mode. This is for PyTorch currently.

        :param model: a PyTorch model
        :param bool is_test: whether in test mode or not.

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
        if self.save_path is not None:
            model_name = os.path.join(self.save_path, model_name)
            if only_param:
                torch.save(model.state_dict(), model_name)
            else:
                torch.save(model, model_name)

    def _load_model(self, model, model_name, only_param=False):
        # TODO: 这个是不是有问题？
        if self.save_path is not None:
            model_path = os.path.join(self.save_path, model_name)
            if only_param:
                states = torch.load(model_path)
            else:
                states = torch.load(model_path).state_dict()
            model.load_state_dict(states)

    def _better_eval_result(self, metrics):
        """Check if the current epoch yields better validation results.

        :return bool value: True means current results on dev set is the best.
        """
        indicator_val = _check_eval_results(metrics, self.metric_key, self.metrics)
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

def _get_value_info(_dict):
    # given a dict value, return information about this dict's value. Return list of str
    strs = []
    for key, value in _dict.items():
        _str = ''
        if isinstance(value, torch.Tensor):
            _str += "\t{}: (1)type:torch.Tensor (2)dtype:{}, (3)shape:{} ".format(key,
                                                                                  value.dtype, value.size())
        elif isinstance(value, np.ndarray):
            _str += "\t{}: (1)type:numpy.ndarray (2)dtype:{}, (3)shape:{} ".format(key,
                                                                                   value.dtype, value.shape)
        else:
            _str += "\t{}: type:{}".format(key, type(value))
        strs.append(_str)
    return strs

def _check_code(dataset, model, losser, metrics, batch_size=DEFAULT_CHECK_BATCH_SIZE,
                dev_data=None, metric_key=None,
                check_level=0):
    # check get_loss 方法
    model_devcie = model.parameters().__next__().device

    batch = Batch(dataset=dataset, batch_size=batch_size, sampler=SequentialSampler())
    for batch_count, (batch_x, batch_y) in enumerate(batch):
        _move_dict_value_to_device(batch_x, batch_y, device=model_devcie)
        # forward check
        if batch_count==0:
            info_str = ""
            input_fields = _get_value_info(batch_x)
            target_fields = _get_value_info(batch_y)
            if len(input_fields)>0:
                info_str += "input fields after batch(if batch size is {}):\n".format(batch_size)
                info_str += "\n".join(input_fields)
                info_str += '\n'
            else:
                raise RuntimeError("There is no input field.")
            if len(target_fields)>0:
                info_str += "target fields after batch(if batch size is {}):\n".format(batch_size)
                info_str += "\n".join(target_fields)
                info_str += '\n'
            else:
                info_str += 'There is no target field.'
            print(info_str)
            _check_forward_error(forward_func=model.forward, dataset=dataset,
                                    batch_x=batch_x, check_level=check_level)

        refined_batch_x = _build_args(model.forward, **batch_x)
        pred_dict = model(**refined_batch_x)
        func_signature = get_func_signature(model.forward)
        if not isinstance(pred_dict, dict):
            raise TypeError(f"The return value of {func_signature} should be `dict`, not `{type(pred_dict)}`.")

        # loss check
        try:
            loss = losser(pred_dict, batch_y)
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
            # TODO: another error raised if CheckError caught
            pre_func_signature = get_func_signature(model.forward)
            _check_loss_evaluate(prev_func_signature=pre_func_signature, func_signature=e.func_signature,
                                 check_res=e.check_res, pred_dict=pred_dict, target_dict=batch_y,
                                 dataset=dataset, check_level=check_level)
        model.zero_grad()
        if batch_count + 1 >= DEFAULT_CHECK_NUM_BATCH:
            break

    if dev_data is not None:
        tester = Tester(data=dataset[:batch_size * DEFAULT_CHECK_NUM_BATCH], model=model, metrics=metrics,
                        batch_size=batch_size, verbose=-1)
        evaluate_results = tester.test()
        _check_eval_results(metrics=evaluate_results, metric_key=metric_key, metric_list=metrics)


def _check_eval_results(metrics, metric_key, metric_list):
    # metrics: tester返回的结果
    # metric_key: 一个用来做筛选的指标，来自Trainer的初始化
    # metric_list: 多个用来做评价的指标，来自Trainer的初始化
    if isinstance(metrics, tuple):
        loss, metrics = metrics

    if isinstance(metrics, dict):
        if len(metrics) == 1:
            # only single metric, just use it
            metric_dict = list(metrics.values())[0]
            metrics_name = list(metrics.keys())[0]
        else:
            metrics_name = metric_list[0].__class__.__name__
            if metrics_name not in metrics:
                raise RuntimeError(f"{metrics_name} is chosen to do validation, but got {metrics}")
            metric_dict = metrics[metrics_name]

        if len(metric_dict) == 1:
            indicator_val, indicator = list(metric_dict.values())[0], list(metric_dict.keys())[0]
        elif len(metric_dict) > 1 and metric_key is None:
            raise RuntimeError(
                f"Got multiple metric keys: {metric_dict}, but metric_key is not set. Which one to use?")
        else:
            # metric_key is set
            if metric_key not in metric_dict:
                raise RuntimeError(f"metric key {metric_key} not found in {metric_dict}")
            indicator_val = metric_dict[metric_key]
    else:
        raise RuntimeError("Invalid metrics type. Expect {}, got {}".format((tuple, dict), type(metrics)))
    return indicator_val

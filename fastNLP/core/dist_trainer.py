r"""
分布式 Trainer
使用步骤
1. 在代码中调用 DistTrainer，类似 Trainer，传入模型和数据等等参数
2. 在命令行中，将 python your_script.py 替换为 python -m torch.distributed.launch --nproc_per_node=N your_script.py
"""
import logging
import os
import time
from datetime import datetime

import contextlib
import torch
import torch.cuda
import torch.distributed as dist
import torch.optim
from torch.serialization import default_restore_location
from pkg_resources import parse_version
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import time

from ._logger import logger, init_logger_dist
from .batch import DataSetIter, BatchIter
from .callback import DistCallbackManager, CallbackException
from .callback import _TesterCallback
from .dataset import DataSet
from .losses import _prepare_losser
from .optimizer import Optimizer
from .utils import _build_args
from .utils import _check_fp16
from .utils import _get_func_signature
from .utils import _move_dict_value_to_device

try:
    from apex import amp
except:
    amp = None

__all__ = [
    'get_local_rank',
    'DistTrainer',
]

def get_local_rank():
    r"""
    返回当前进程的 local rank， 0 到 N-1 ，N为当前分布式总进程数
    """
    if 'LOCAL_RANK' in os.environ:
        return int(os.environ['LOCAL_RANK'])
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--local_rank', type=int)
    args, _ = parser.parse_known_args()
    if 'local_rank' in args and args.local_rank:
        os.environ['LOCAL_RANK'] = str(args.local_rank) # for multiple calls for this function
        return args.local_rank
    raise RuntimeError('Please use "python -m torch.distributed.launch --nproc_per_node=N train_script.py')


class DistTrainer():
    r"""
    分布式的 Trainer，支持分布式训练和混合精度的训练。具体实现原理请阅读 pytorch 官方文档。

    Note: 使用分布式 Trainer 时会同时有多个进程执行训练代码。因此将单进程的训练代码改为多进程之前，
    请仔细检查，确保训练代码中的同步和互斥操作能正确执行（如模型保持，打印日志等）
    """
    def __init__(self, train_data, model, optimizer=None, loss=None,
                 callbacks_all=None, callbacks_master=None,
                 batch_size_per_gpu=8, n_epochs=1,
                 num_workers=1, drop_last=False,
                 dev_data=None, metrics=None, metric_key=None,
                 update_every=1, print_every=10, validate_every=-1,
                 save_path=None, device='auto',
                 fp16='', use_tqdm=True):
        r"""

        :param train_data: 训练集， :class:`~fastNLP.DataSet` 类型。
        :param nn.modules model: 待训练的模型
        :param optimizer: `torch.optim.Optimizer` 优化器。如果为None，则Trainer使用默认的Adam(model.parameters(), lr=4e-3)这个优化器
        :param loss: 使用的 :class:`~fastNLP.core.losses.LossBase` 对象。当为None时，默认使用 :class:`~fastNLP.LossInForward`
        :param list callbacks_all: 用于在train过程中起调节作用的回调函数，作用于所有训练进程中。
            可使用的callback参见 :mod:`callback模块 <fastNLP.core.callback>`
        :param list callbacks_master: 用于在train过程中起调节作用的回调函数，只作用于其中一个进程（ Master 进程）。
            可使用的callback参见 :mod:`callback模块 <fastNLP.core.callback>`
        :param int batch_size_per_gpu: 训练时，每个进程的 batch 大小。
        :param int n_epochs: 需要优化迭代多少次。
        :param num_workers: int, 有多少个线程来进行数据pad处理。
        :param drop_last: 如果最后一个batch没有正好为batch_size这么多数据，就扔掉最后一个batch
        :param dev_data: 用于做验证的DataSet， :class:`~fastNLP.DataSet` 类型。
        :param metrics: 验证的评估函数。可以只使用一个 :class:`Metric<fastNLP.core.metrics.MetricBase>` ，
            也可以使用多个 :class:`Metric<fastNLP.core.metrics.MetricBase>` ，通过列表传入。
            如验证时取得了更好的验证结果(如果有多个Metric，以列表中第一个Metric为准)，且save_path不为None，
            则保存当前模型。Metric种类详见 :mod:`metrics模块 <fastNLP.core.metrics>` 。仅在传入dev_data时有效。
        :param str,None metric_key:  :class:`Metric<fastNLP.core.metrics.MetricBase>` 有时会有多个指标，
            比如 :class:`~fastNLP.core.metrics.SpanFPreRecMetric` 中包含了'f', 'pre', 'rec'。此时需
            要指定以哪个指标为准。另外有些指标是越小效果越好，比如语言模型的困惑度，这种情况下，在key前面增加一个'-'来表
            明验证时，值越小越好(比如: "-ppl")。仅在传入dev_data时有效。
        :param update_every: int, 多少步更新一次梯度。用于希望累计梯度的场景，比如需要128的batch_size, 但是直接设为128
            会导致内存不足，通过设置batch_size=32, update_every=4达到目的。当optimizer为None时，该参数无效。
        :param int print_every: 多少次反向传播更新tqdm显示的loss; 如果use_tqdm=False, 则多少次反向传播打印loss。
        :param int validate_every: 多少个step在验证集上验证一次; 如果为-1，则每个epoch结束验证一次。仅在传入dev_data时有效。
        :param str,None save_path: 将模型保存路径，如果路径不存在，将自动创建文件夹。如果为None，则不保存模型。如果dev_data为None，则保存
            最后一次迭代的模型。保存的时候不仅保存了参数，还保存了模型结构。即便使用DataParallel，这里也只保存模型。
        :param str device: 指定 device，可以是 gpu，cpu 或 auto
        :param str fp16: 指定半精度训练的优化等级，可为 O1，O2 或 O3，若为空字符串则不使用半精度。
        :param bool use_tqdm: 是否使用tqdm来显示训练进度; 如果为False，则将loss打印在终端中。
        """
        assert device in ['auto', 'cuda', 'cpu'], "Please set correct device in [auto', 'cuda', 'cpu']"
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # init distributed
        if device == 'cuda':
            torch.cuda.set_device(get_local_rank())
            self.device = torch.device("cuda", get_local_rank())
        else:
            self.device = torch.device(device)

        init_logger_dist()

        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank() # unique id for each process

        self.train_data = train_data
        self.batch_size_per_gpu = int(batch_size_per_gpu)
        self.n_epochs = int(n_epochs)
        self.num_data_workers = int(num_workers)
        self.drop_last = drop_last
        self.update_every = int(update_every)
        self.print_every = int(print_every)
        self.validate_every = int(validate_every)
        self.save_path = save_path
        self.losser = _prepare_losser(loss)
        self.fp16 = fp16
        self.local_rank = get_local_rank()
        self._forward_func = model.forward
        self.callback_manager = DistCallbackManager(
            env={"trainer": self}, callbacks_all=callbacks_all,
            callbacks_master=callbacks_master)
        self.test_manager = DistCallbackManager(env={'trainer': self})
        self.metric_key = metric_key
        self.use_tqdm = use_tqdm

        model.to(self.device)
        optimizer = self._get_optimizer(optimizer)

        # init fp16, must before DataParallel init
        if len(self.fp16):
            assert isinstance(self.fp16, str), "Please set Apex AMP optimization level selected in ['O0', 'O1', 'O2', 'O3']"
            _check_fp16()
            assert device == 'cuda', "Amp requires cuda device"
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.fp16)

        # init DataParallel
        if parse_version(torch.__version__)>=parse_version('1.1'):
            self.ddp_model = DDP(model, device_ids=[self.local_rank],
                             output_device=self.local_rank, find_unused_parameters=True)
        else:
            self.ddp_model = DDP(model, device_ids=[self.local_rank],
                             output_device=self.local_rank)
        self.model = self.ddp_model.module

        self.optimizer = optimizer
        self.sampler = DistributedSampler(self.train_data)
        self.data_iterator = self._get_data_iter(self.train_data)
        self.batch_size = self.world_size * self.batch_size_per_gpu
        self.n_steps = self._get_n_steps()

        # for evaluation, only run eval on master proc
        if dev_data and metrics:
            cb = _TesterCallback(
                dev_data, model, metrics,
                batch_size=batch_size_per_gpu, num_workers=num_workers)
            self.test_manager.add_callback([cb], master=True)

        # Setup logging
        # 同步start_time
        sync_time = torch.tensor(time.time(), dtype=torch.double).to(self.device)
        dist.broadcast(sync_time, src=0)
        self.start_time = datetime.fromtimestamp(sync_time.item()).strftime('%Y-%m-%d-%H-%M-%S-%f')
        # print('sync_time: {}, start_time: {}'.format(sync_time, self.start_time))

        if self.save_path:
            self.cp_save_path = self.save_path
        else:
            self.cp_save_path = None
        # use INFO in the master, WARN for others
        self.logger = logger
        self.logger.info("Setup Distributed Trainer")
        self.logger.warning("Process pid: {}, rank: {}, local rank: {}, device: {}, fp16: {}".format(
                        os.getpid(), self.rank, self.local_rank, self.device, self.fp16 if self.fp16 else False))
        self.logger.info("Num of processes: {}".format(self.world_size))
        self.logger.info("Use device: {}".format(device))
        self.logger.info("Training with fp16: {}, optimization level: {}".format(
                        len(self.fp16) > 0, self.fp16 if self.fp16 else None))

    def _maybe_no_sync(self):
        """
        Whenever *samples* contains more than one mini-batch, we
        want to accumulate gradients locally and only call
        all-reduce in the last backwards pass.
        """
        i = self.step % self.update_every
        if (
                self.world_size > 1
                and hasattr(self.ddp_model, "no_sync")
                and i != 0
        ):
            return self.ddp_model.no_sync()
        else:
            return contextlib.ExitStack()  # dummy contextmanager

    def _get_n_steps(self):
        return len(self.data_iterator) * self.n_epochs

    def _get_data_iter(self, dataset):
        if isinstance(dataset, DataSet):
            return DataSetIter(dataset=dataset, batch_size=self.batch_size_per_gpu, sampler=self.sampler,
                               num_workers=self.num_data_workers, drop_last=self.drop_last)
        elif isinstance(dataset, BatchIter):
            return dataset
        else:
            raise TypeError("train_data type {} not support".format(type(dataset)))

    def _get_optimizer(self, optimizer):
        if isinstance(optimizer, torch.optim.Optimizer):
            return optimizer
        elif isinstance(optimizer, Optimizer):
            return optimizer.construct_from_pytorch(self.ddp_model.parameters())
        elif optimizer is None:
            return torch.optim.Adam(self.ddp_model.parameters(), lr=4e-3)
        else:
            raise TypeError("optimizer can only be torch.optim.Optimizer type, not {}.".format(type(optimizer)))

    @property
    def is_master(self):
        r"""是否是主进程"""
        return self.rank == 0

    def train(self, load_best_model=True, on_exception='auto'):
        r"""
        使用该函数使Trainer开始训练。

        :param str on_exception: 在训练过程遭遇exception，并被 :py:class:Callback 的on_exception()处理后，是否继续抛出异常。
                支持'ignore','raise', 'auto': 'ignore'将捕获异常，写在Trainer.train()后面的代码将继续运行; 'raise'将异常抛出;
                'auto'将ignore以下两种Exception: CallbackException与KeyboardInterrupt, raise其它exception.
        :return dict: 返回一个字典类型的数据,
                内含以下内容::

                    seconds: float, 表示训练时长
                    以下三个内容只有在提供了dev_data的情况下会有。
                    best_eval: Dict of Dict, 表示evaluation的结果。第一层的key为Metric的名称，
                                第二层的key为具体的Metric
                    best_epoch: int，在第几个epoch取得的最佳值
                    best_step: int, 在第几个step(batch)更新取得的最佳值

        """
        try:
            self.logger.info("###### Training epochs started ######")
            self.logger.info('Total epochs: %d'% self.n_epochs)
            self.logger.info('Total steps: %d'% self.n_steps)
            self.logger.info('Num instances per GPU: %d'% self.batch_size_per_gpu)
            self.logger.info('Num of steps per update: %d' % self.update_every)
            self.logger.info('Total batch_size: %d'%
                             (self.batch_size_per_gpu * dist.get_world_size() * self.update_every))
            self.logger.info('Total num of samples: %d'% len(self.train_data))
            self.logger.info("Num of callbacks for all workers: {}".format(
                                len(self.callback_manager.callbacks_all)))
            self.logger.info("Num of callbacks for master workers: {}".format(
                                len(self.callback_manager.callbacks_master)))
            self.logger.info("Callbacks for all workers: {}".format(
                    [repr(cb) for cb in self.callback_manager.callbacks_all]))
            self.logger.info("Callbacks for master workers: {}".format(
                    [repr(cb) for cb in self.callback_manager.callbacks_master]))

            start_time = time.time()
            results = {}
            if self.n_epochs <= 0:
                self.logger.info("Training epoch is {}, nothing was done.".format(self.n_epochs))
                results['seconds'] = 0.
                return results

            try:
                self.callback_manager.on_train_begin()
                self._train()
                self.callback_manager.on_train_end()

            except BaseException as e:
                self.callback_manager.on_exception(e)
                if on_exception == 'auto':
                    if not isinstance(e, (CallbackException, KeyboardInterrupt)):
                        raise e
                    else:
                        self.logger.info('Catch {}, ignored.'.format(e.__class__.__name__))
                elif on_exception == 'raise':
                    raise e

            results['seconds'] = round(time.time() - start_time, 2)
            self.logger.info("###### Train finished ######")
            self.logger.info('Total train time: {} seconds.'. format(results['seconds']))
            if load_best_model and self.cp_save_path and len(self.test_manager.callbacks):
                self.load_check_point(self._best_save_name())
        finally:
            pass
        dist.barrier()
        return results

    def _train(self):
        dist.barrier()
        if not self.use_tqdm:
            from .utils import _pseudo_tqdm as inner_tqdm
        else:
            inner_tqdm = tqdm

        self.step = 0
        self.epoch = 0
        self.pbar = inner_tqdm(total=self.n_steps, postfix='loss:{0:<6.5f}',
                        leave=False, dynamic_ncols=True, disable=not self.is_master)
        pbar = self.pbar
        avg_loss = 0
        data_iterator = self.data_iterator
        self.ddp_model.zero_grad()
        for epoch in range(1, self.n_epochs + 1):
            self.epoch = epoch
            pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.n_epochs))
            # early stopping
            self.callback_manager.on_epoch_begin()
            for batch_x, batch_y in data_iterator:
                self.step += 1
                self.ddp_model.train()
                _move_dict_value_to_device(batch_x, batch_y, device=self.device)
                indices = data_iterator.get_batch_indices()
                # negative sampling; replace unknown; re-weight batch_y
                self.callback_manager.on_batch_begin(batch_x, batch_y, indices)
                prediction = self._data_forward(self.ddp_model, batch_x)

                # edit prediction
                self.callback_manager.on_loss_begin(batch_y, prediction)
                loss = self._compute_loss(prediction, batch_y)
                if self.update_every > 1:
                    loss = loss / self.update_every
                avg_loss += loss.item()

                # Is loss NaN or inf? requires_grad = False
                self.callback_manager.on_backward_begin(loss)

                # with self._maybe_no_sync():
                if self.fp16:
                    with amp.scale_loss(loss, self.optimizer) as scale_loss:
                        scale_loss.backward()
                else:
                    loss.backward()

                self.callback_manager.on_backward_end()

                self._update()
                self.callback_manager.on_step_end()

                if self.step % self.print_every == 0:
                    avg_loss = float(avg_loss) / self.print_every
                    print_output = "loss:{:<6.5f}".format(avg_loss)
                    pbar.update(self.print_every)
                    pbar.set_postfix_str(print_output)
                    avg_loss = 0

                self.callback_manager.on_batch_end()

                if (self.validate_every > 0 and self.step % self.validate_every == 0):
                    self._do_validation()

            # ================= mini-batch end ==================== #
            if self.validate_every < 0:
                self._do_validation()

            # lr decay; early stopping
            self.callback_manager.on_epoch_end()
        # =============== epochs end =================== #
        pbar.close()
        self.pbar = None
    # ============ tqdm end ============== #

    def _update(self):
        r"""Perform weight update on a model.

        """
        if self.step % self.update_every == 0:
            self.optimizer.step()
            self.ddp_model.zero_grad()

    def _data_forward(self, network, x):
        x = _build_args(self._forward_func, **x)
        y = network(**x)
        if not isinstance(y, dict):
            raise TypeError(
                f"The return value of {_get_func_signature(self._forward_func)} should be dict, got {type(y)}.")
        return y

    def _compute_loss(self, predict, truth):
        r"""Compute loss given prediction and ground truth.

        :param predict: prediction dict, produced by model.forward
        :param truth: ground truth dict, produced by batch_y
        :return: a scalar
        """
        loss = self.losser(predict, truth)
        if self.update_every > 1:
            loss = loss / self.update_every
        if loss.dim() > 0:
            loss = loss.mean()
        return loss

    def save_check_point(self, name=None, only_params=False):
        r"""保存当前模型"""
        # only master save models
        if name is None:
            name = 'checkpoint-{}.bin'.format(self.step)
        os.makedirs(self.cp_save_path, exist_ok=True)
        path = os.path.join(self.cp_save_path, name)
        self.logger.info("Save checkpoint to {}".format(path))
        model_to_save = self.ddp_model.module
        if only_params:
            model_to_save = model_to_save.state_dict()
        if self.is_master:
            torch.save(model_to_save, path)

    def load_check_point(self, name):
        path = os.path.join(self.cp_save_path, name)
        self.logger.info('reload best model from %s', path)
        model_load = torch.load(
            path,
            map_location=lambda s, l: default_restore_location(s, "cpu"))
        if not isinstance(model_load, dict):
            model_load = model_load.state_dict()
        self.model.load_state_dict(model_load)

    def _best_save_name(self, auto_fix=True):
        best_name = "best_" + "_".join([self.model.__class__.__name__, str(self.metric_key), self.start_time])
        return best_name

    def _do_validation(self):
        with self.ddp_model.no_sync():
            # 因为模型参数不更新，可以关闭同步
            self.callback_manager.on_valid_begin()
            eval_res = self.test_manager.on_valid_begin()
            eval_res = list(filter(lambda x: x is not None, eval_res))
            if len(eval_res):
                eval_res, is_better = list(zip(*eval_res))
                eval_res = eval_res[0]
                is_better = is_better[0]
            else:
                eval_res, is_better = None, None
            if self.metric_key is None and eval_res is not None:
                eval_res0 = list(eval_res.values())[0]
                self.metric_key = list(eval_res0.keys())[0]
            # logger.info('{}, {}'.format(eval_res, is_better))
            # save better model on master node
            if is_better is not None and self.cp_save_path:
                if is_better:
                    self.save_check_point(self._best_save_name(), only_params=False)
            dist.barrier()

            if not self.is_master and self.metric_key is None:
                # 主进程自动得到了metric_key，而其它进程没有
                prefix = 'best_' + self.model.__class__.__name__
                suffix = self.start_time
                fn_list = os.listdir(self.cp_save_path)
                fn_list = [fn for fn in fn_list if fn.startswith(prefix) and fn.endswith(suffix)]
                if len(fn_list) == 1:
                    best_name = fn_list[0]
                    self.metric_key = best_name[len(prefix):-len(suffix)].strip('_')
            # print('RANK {} metric_key {}'.format(self.rank, self.metric_key))
            self.callback_manager.on_valid_end(
                eval_res, self.metric_key, self.optimizer, is_better)
            self.ddp_model.train()

    def close(self):
        r"""关闭Trainer，销毁进程"""
        dist.destroy_process_group()

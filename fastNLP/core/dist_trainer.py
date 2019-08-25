"""
正在开发中的分布式训练代码
"""
import torch
import torch.cuda
import torch.optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from tqdm import tqdm
import time
from datetime import datetime, timedelta
from functools import partial

from .batch import DataSetIter, BatchIter
from .callback import DistCallbackManager, CallbackException, TesterCallback
from .dataset import DataSet
from .losses import _prepare_losser
from .optimizer import Optimizer
from .utils import _build_args
from .utils import _move_dict_value_to_device
from .utils import _get_func_signature
from  ._logger import logger
import logging
from pkg_resources import parse_version

__all__ = [
    'get_local_rank',
    'DistTrainer',
]


def get_local_rank():
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
    """
    Distributed Trainer that support distributed and mixed precision training
    """
    def __init__(self, train_data, model, optimizer=None, loss=None,
                 callbacks_all=None, callbacks_master=None,
                 batch_size_per_gpu=8, n_epochs=1,
                 num_workers=1, drop_last=False,
                 dev_data=None, metrics=None, metric_key=None,
                 update_every=1, print_every=10, validate_every=-1,
                 save_every=-1, save_path=None, device='auto',
                 fp16='', backend=None, init_method=None):

        assert device in ['auto', 'cuda', 'cpu'], "Please set correct device in [auto', 'cuda', 'cpu']"
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if backend is None:
            backend = 'nccl' if device == 'cuda' else 'gloo'

        # init distributed
        if device == 'cuda':
            torch.cuda.set_device(get_local_rank())
            self.device = torch.device("cuda", get_local_rank())
        else:
            self.device = torch.device(device)

        dist.init_process_group(backend=backend, init_method=init_method)
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank() # unique id for each process

        self.model = model
        self.train_data = train_data
        self.batch_size_per_gpu = int(batch_size_per_gpu)
        self.n_epochs = int(n_epochs)
        self.num_data_workers = int(num_workers)
        self.drop_last = drop_last
        self.update_every = int(update_every)
        self.print_every = int(print_every)
        self.validate_every = int(validate_every)
        self.save_every = int(save_every)
        self.save_path = save_path
        self.losser = _prepare_losser(loss)
        self.fp16 = fp16
        self.init_method = init_method
        self.backend = backend
        self.local_rank = get_local_rank()
        self._forward_func = model.forward
        self.callback_manager = DistCallbackManager(
            env={"trainer": self}, callbacks_all=callbacks_all,
            callbacks_master=callbacks_master)
        self.metric_key = metric_key

        model.to(self.device)
        optimizer = self._get_optimizer(optimizer)

        # init fp16, must before DataParallel init
        if len(self.fp16):
            assert isinstance(self.fp16, str), "Please set Apex AMP optimization level selected in ['O0', 'O1', 'O2', 'O3']"
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
            assert device == 'cuda', "Amp requires cuda device"
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.fp16)

        # init DataParallel
        if parse_version(torch.__version__)>=parse_version('1.1'):
            self.model = DDP(model, device_ids=[self.local_rank],
                             output_device=self.local_rank, find_unused_parameters=True)
        else:
            self.model = DDP(model, device_ids=[self.local_rank],
                             output_device=self.local_rank)

        self.optimizer = optimizer
        self.sampler = DistributedSampler(self.train_data)
        self.data_iterator = self._get_data_iter(self.train_data)
        self.n_steps = self._get_n_steps()

        # for evaluation, only run eval on master proc
        if dev_data and metrics:
            cb = TesterCallback(
                dev_data, model, metrics,
                batch_size=batch_size_per_gpu, num_workers=num_workers)
            self.callback_manager.add_callback([cb], master=True)

        # Setup logging
        dist.barrier()
        self.start_time = datetime.now().strftime('%m_%d_%Y-%H_%M')
        if self.save_path:
            self.cp_save_path = os.path.join(self.save_path, 'checkpoints', self.start_time)
        else:
            self.cp_save_path = None

        # use INFO in the master, WARN for others
        logger.setLevel(logging.INFO  if self.is_master else logging.WARNING)
        self.logger = logger
        self.logger.info("Setup Distributed Trainer")
        self.logger.warning("Process pid: {}, rank: {}, local rank: {}, device: {}, fp16: {}".format(
                        os.getpid(), self.rank, self.local_rank, self.device, self.fp16 if self.fp16 else False))
        self.logger.info("Num of processes: {}".format(self.world_size))
        self.logger.info("Use device: {}".format(device))
        self.logger.info("Training with fp16: {}, optimization level: {}".format(
                        len(self.fp16) > 0, self.fp16 if self.fp16 else None))

    def _get_n_steps(self):
        batch_size = self.world_size * self.batch_size_per_gpu
        return (len(self.train_data) // batch_size + int(
            len(self.train_data) % batch_size != 0)) * int(self.drop_last == 0) * self.n_epochs

    def _get_data_iter(self, dataset):
        if isinstance(dataset, DataSet):
            return DataSetIter(
                dataset=dataset, batch_size=self.batch_size_per_gpu,
                num_workers=self.num_data_workers, sampler=self.sampler,
                drop_last=self.drop_last
            )
        elif isinstance(dataset, BatchIter):
            return dataset
        else:
            raise TypeError("train_data type {} not support".format(type(dataset)))

    def _get_optimizer(self, optimizer):
        if isinstance(optimizer, torch.optim.Optimizer):
            return optimizer
        elif isinstance(optimizer, Optimizer):
            return optimizer.construct_from_pytorch(self.model.parameters())
        elif optimizer is None:
            return torch.optim.Adam(self.model.parameters(), lr=4e-3)
        else:
            raise TypeError("optimizer can only be torch.optim.Optimizer type, not {}.".format(type(optimizer)))

    @property
    def is_master(self):
        return self.rank == 0

    def train(self, on_exception='auto'):
        try:
            self.logger.info("###### Training epochs started ######")
            self.logger.info('Total epochs: %d'% self.n_epochs)
            self.logger.info('Total steps: %d'% self.n_steps)
            self.logger.info('Num instances per GPU %d'% self.batch_size_per_gpu)
            self.logger.info('Total batch_size: %d'% self.batch_size_per_gpu * dist.get_world_size())
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
            return results
        finally:
            self.close()

    def _train(self):
        if self.fp16:
            # skip check, done in __init__()
            from apex import amp
        self.step = 0
        self.epoch = 0
        self.pbar = tqdm(total=self.n_steps, postfix='loss:{0:<6.5f}',
                        leave=False, dynamic_ncols=True, disable=not self.is_master)
        pbar = self.pbar
        avg_loss = 0
        data_iterator = self.data_iterator
        self.model.zero_grad()
        for epoch in range(1, self.n_epochs + 1):
            self.epoch = epoch
            pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.n_epochs))
            # early stopping
            self.callback_manager.on_epoch_begin()
            for batch_x, batch_y in data_iterator:
                self.model.train()
                self.step += 1
                _move_dict_value_to_device(batch_x, batch_y, device=self.device)
                indices = data_iterator.get_batch_indices()
                # negative sampling; replace unknown; re-weight batch_y
                self.callback_manager.on_batch_begin(batch_x, batch_y, indices)
                prediction = self._data_forward(self.model, batch_x)

                # edit prediction
                self.callback_manager.on_loss_begin(batch_y, prediction)
                loss = self._compute_loss(prediction, batch_y)
                avg_loss += loss.item()

                # Is loss NaN or inf? requires_grad = False
                self.callback_manager.on_backward_begin(loss)

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

                if self.cp_save_path and \
                        self.save_every > 0 and \
                        self.step % self.save_every == 0:
                    self.save_check_point()

            # ================= mini-batch end ==================== #
            if self.validate_every < 0:
                self._do_validation()

        if self.save_every < 0 and self.cp_save_path:
            self.save_check_point()
            # lr decay; early stopping
            self.callback_manager.on_epoch_end()
        # =============== epochs end =================== #
        pbar.close()
        self.pbar = None
    # ============ tqdm end ============== #

    def _update(self):
        """Perform weight update on a model.

        """
        if self.step % self.update_every == 0:
            self.optimizer.step()
            self.model.zero_grad()

    def _data_forward(self, network, x):
        x = _build_args(self._forward_func, **x)
        y = network(**x)
        if not isinstance(y, dict):
            raise TypeError(
                f"The return value of {_get_func_signature(self._forward_func)} should be dict, got {type(y)}.")
        return y

    def _compute_loss(self, predict, truth):
        """Compute loss given prediction and ground truth.

        :param predict: prediction dict, produced by model.forward
        :param truth: ground truth dict, produced by batch_y
        :return: a scalar
        """
        loss = self.losser(predict, truth)
        if self.update_every > 1:
            loss = loss / self.update_every
        return loss.mean()

    def save_check_point(self, only_params=False):
        # only master save models
        if self.is_master:
            os.makedirs(self.cp_save_path, exist_ok=True)
            path = os.path.join(self.cp_save_path, 'checkpoint-{}.bin'.format(self.step))
            self.logger.info("Save checkpoint to {}".format(path))
            model_to_save = self.model.module
            if only_params:
                model_to_save = model_to_save.state_dict()
            torch.save(model_to_save, path)

    def _do_validation(self):
        self.callback_manager.on_valid_begin()
        eval_res = self.callback_manager.on_validation()
        eval_res = list(filter(lambda x: x is not None, eval_res))
        if len(eval_res):
            eval_res, is_better = list(zip(*eval_res))
        else:
            eval_res, is_better = None, None
        self.callback_manager.on_valid_end(
            eval_res, self.metric_key, self.optimizer, is_better)
        dist.barrier()

    def close(self):
        dist.destroy_process_group()

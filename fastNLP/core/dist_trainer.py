import torch
import torch.cuda
import torch.optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from tqdm import tqdm
import logging
import time
from datetime import datetime, timedelta

from .batch import DataSetIter, BatchIter
from .callback import CallbackManager, CallbackException
from .dataset import DataSet
from .losses import _prepare_losser
from .optimizer import Optimizer
from .utils import _build_args
from .utils import _move_dict_value_to_device
from .utils import _get_func_signature

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
    raise RuntimeError('Please use "python -m torch.distributed.launch train_script.py')


class DistTrainer():
    def __init__(self, model, train_data, optimizer, loss, callbacks=None,
                 batch_size_per_gpu=8, n_epochs=1,
                 num_workers=1, drop_last=False,
                 update_every=1, print_every=10, validate_every=-1,
                 save_every=-1, save_path=None,
                 logging_level=logging.INFO,
                 fp16='', backend='nccl', init_method=None):
        self.model = model
        self.train_data = train_data
        self.batch_size_per_gpu = int(batch_size_per_gpu)
        self.n_epochs = int(n_epochs)
        self.num_workers = int(num_workers)
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
        self.callback_manager = CallbackManager(env={"trainer": self}, callbacks=callbacks)
        self._forward_func = model.forward

        assert torch.cuda.is_available(), "Distributed Trainer requires cuda to be enabled."
        # init distributed
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        dist.init_process_group(backend=self.backend, init_method=self.init_method)
        model.to(self.device)
        optimizer = self.get_optimizer(optimizer)

        # init fp16, must before DataParallel init
        if len(self.fp16):
            assert isinstance(self.fp16, str), "Please set Apex AMP optimization level selected in ['O0', 'O1', 'O2', 'O3']"
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.fp16)

        # init DataParallel
        self.model = DDP(model, device_ids=[self.local_rank],
                         output_device=self.local_rank)
        self.optimizer = optimizer
        self.world_size = dist.get_world_size()
        self.rank = dist.get_rank() # unique id for each process
        self.sampler = DistributedSampler(self.train_data)
        self.data_iterator = self.get_data_iter(self.train_data)
        self.n_steps = self.get_n_steps()

        # Setup logging
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S',
                            level=logging_level)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Process pid: {}, rank: {}, local rank: {}, device: {}, fp16: {}".format(
                        os.getpid(), self.rank, self.local_rank, self.device, self.fp16 if self.fp16 else False))
        if self.is_master:
            self.logger.info('Total epochs: %d'% self.n_epochs)
            self.logger.info('Total steps: %d'% self.n_steps)
            self.logger.info('Num instances per GPU %d'% self.batch_size_per_gpu)
            self.logger.info('Total batch_size: %d'% self.batch_size_per_gpu * dist.get_world_size())
            self.logger.info('Total num of samples: %d'% len(self.train_data))
            self.logger.info("Num of callbacks: {}".format(len(self.callback_manager.callbacks)))
            self.logger.info(
                "Use callbacks: {}".format([repr(cb) for cb in self.callback_manager.callbacks]))

            # only master process save model
            if self.save_path:
                self.save_path = os.path.join(
                    self.save_path,
                    datetime.now().strftime('%m_%d_%y-%H_%M_%S')+'-'+str(os.getpid()))

    def get_n_steps(self):
        batch_size = self.world_size * self.batch_size_per_gpu
        return (len(self.train_data) // batch_size + int(
            len(self.train_data) % batch_size != 0)) * int(self.drop_last == 0) * self.n_epochs

    def get_data_iter(self, dataset):
        if isinstance(dataset, DataSet):
            return DataSetIter(
                dataset=dataset, batch_size=self.batch_size_per_gpu,
                num_workers=self.num_workers, sampler=self.sampler,
                drop_last=self.drop_last
            )
        elif isinstance(dataset, BatchIter):
            return dataset
        else:
            raise TypeError("train_data type {} not support".format(type(dataset)))

    def get_optimizer(self, optimizer):
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
        start_time = time.time()
        results = {}
        if self.n_epochs <= 0:
            if self.is_master:
                self.logger.info("Training epoch is {}, nothing was done.".format(self.n_epochs))
            results['seconds'] = 0.
            return results

        if self.is_master:
            self.logger.info("###### Training epochs started ######")

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
        if self.is_master:
            self.logger.info("###### Train finished ######")
            self.logger.info('Total train time: {} seconds.'. format(results['seconds']))
        return results

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

                if ((self.validate_every > 0 and self.step % self.validate_every == 0) or
                    (self.validate_every < 0 and self.step % len(data_iterator) == 0)):
                    eval_str = "Evaluation at Epoch {}/{}. Step:{}/{}. ".format(epoch, self.n_epochs, self.step,
                                                                                self.n_steps)
                    if self.is_master:
                        self.logger.info(eval_str)
                    self.callback_manager.on_validation()
                    dist.barrier()

                if self.save_path and \
                        self.save_every > 0 and \
                        self.step % self.save_every == 0:
                    self.save_check_point()

            # ================= mini-batch end ==================== #
        if self.save_path and self.save_every < 0:
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
        if self.is_master:
            if not os.path.exists(self.save_path):
                os.makedirs(self.save_path)
            path = os.path.join(self.save_path, 'checkpoint-{}.bin'.format(self.step))
            self.logger.info("Save checkpoint to {}".format(path))
            model_to_save = self.model.module
            if only_params:
                model_to_save = model_to_save.state_dict()
            torch.save(model_to_save, path)
        dist.barrier()

    def close(self):
        dist.destroy_process_group()

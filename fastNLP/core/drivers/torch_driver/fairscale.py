import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

from fastNLP.envs.imports import _NEED_IMPORT_FAIRSCALE

if _NEED_IMPORT_FAIRSCALE:
    import torch
    import torch.distributed as dist
    from fairscale.optim import OSS
    from fairscale.nn import ShardedDataParallel
    from fairscale.nn import FullyShardedDataParallel
    from fairscale.optim.grad_scaler import ShardedGradScaler
    from torch.nn.parallel import DistributedDataParallel

from fastNLP.envs import FASTNLP_DISTRIBUTED_CHECK
from ....envs.distributed import all_rank_call_context
from ...log import logger
from .ddp import TorchDDPDriver
from .torch_driver import TorchDriver
from .utils import (_build_fp16_env, _DDPWrappingModel,
                    optimizer_state_to_device)

__all__ = ['FairScaleDriver']


class FairScaleDriver(TorchDDPDriver):
    r"""实现 ``fairscale`` 功能的 ``Driver``。

    :param model: 传入给 :class:`.Trainer` 的 ``model`` 参数。
    :param parallel_device: 用于分布式训练的 ``gpu`` 设备。
    :param is_pull_by_torch_run: 标志当前的脚本的启动是否由 ``python -m
        torch.distributed.launch`` 启动的。
    :param fp16: 是否开启 fp16 训练。
    :param fairscale_kwargs:

        * *fs_type* -- 使用 ``fairscale`` 进行分布式训练的模式，包括 ``['ddp',
          'sdp', 'fsdp']`` 三种模式，分别代表 ``DistributedDataParallel``、
          ``ShardedDataParallel`` 和 ``FullyShardedDataParallel``。
        * *oss_kwargs* --
        * *sdp_kwargs* --
        * *fsdp_kwargs* --
        * *ddp_kwargs* --
        * *set_grad_to_none* -- 是否在训练过程中在每一次 optimizer 更新后将 grad
          置为 ``None``
        * *non_blocking* -- 表示用于 :meth:`torch.Tensor.to` 方法的参数
          non_blocking
        * *gradscaler_kwargs* -- 用于 ``fp16=True`` 时，提供给 :class:`torch.\
          amp.cuda.GradScaler` 的参数
    :kwargs:
        * *model_wo_auto_param_call* (``bool``) -- 是否关闭在训练时调用我们的
          ``auto_param_call`` 函数来自动匹配 batch 和前向函数的参数的行为

        .. note::

            关于该参数的详细说明，请参见 :class:`.Trainer` 和 :func:`~fastNLP.\
            core.auto_param_call`。

        * *output_from_new_proc* (``str``) -- 应当为一个字符串，表示在多进程的
          driver 中其它进程的输出流应当被做如何处理；其值应当为以下之一： ``["all",
          "ignore", "only_error"]``，分别代表 *全部输出*、 *全部忽略* 和 *仅输出错
          误* ，而 rank0 的 **所有信息** 都将被打印出来；当该参数的值不是以上值时，
          该值应当表示一个文件夹的名字，我们会将其他 rank 的输出流重定向到 log 文件
          中，然后将 log 文件保存在通过该参数值设定的文件夹中；默认为
          ``"only_error"``；
    """

    def __init__(self,
                 model,
                 parallel_device: Union[List['torch.device'], 'torch.device'],
                 is_pull_by_torch_run=False,
                 fp16: bool = False,
                 fairscale_kwargs: Optional[Dict] = None,
                 **kwargs):
        assert _NEED_IMPORT_FAIRSCALE, 'fairscale is not imported.'
        assert not dist.is_initialized(), 'FairScaleDriver does not support ' \
                                          'initialize distributed by user.'
        self._fairscale_kwargs = fairscale_kwargs if fairscale_kwargs is not None else {}
        self.fs_type = self._fairscale_kwargs.get('fs_type',
                                                  'sdp')  # ddp, sdp, fsdp
        if self.fs_type == 'fsdp':
            self._fairscale_kwargs[
                'set_grad_to_none'] = self._fairscale_kwargs.get(
                    'set_grad_to_none', True)
        # 将最顶上的进行初始化
        kwargs.pop('torch_kwargs', None)
        TorchDriver.__init__(
            self,
            model=model,
            fp16=False,
            torch_kwargs=self._fairscale_kwargs,
            **kwargs)
        self.is_pull_by_torch_run = is_pull_by_torch_run
        assert self.fs_type in ['ddp', 'sdp', 'fsdp']
        # 仅在 ddp 和 sdp 下有使用到
        self._oss_kwargs = self._fairscale_kwargs.get('oss_kwargs', {})
        self._sdp_kwargs = self._fairscale_kwargs.get('sdp_kwargs', {})
        self._fsdp_kwargs = self._fairscale_kwargs.get('fsdp_kwargs', {})
        self._ddp_kwargs = self._fairscale_kwargs.get('ddp_kwargs', {})

        if self.fs_type == 'ddp' or fp16 is False:
            self.auto_cast, _grad_scaler = _build_fp16_env(dummy=not fp16)
            self.grad_scaler = _grad_scaler(
                **self._fairscale_kwargs.get('gradscaler_kwargs', {}))
        else:
            self.auto_cast, self.grad_scaler = torch.cuda.amp.autocast, \
                ShardedGradScaler(
                    **self._fairscale_kwargs.get('gradscaler_kwargs', {}))

        self.parallel_device = parallel_device
        if is_pull_by_torch_run:
            self.model_device = parallel_device
        else:
            self.model_device = parallel_device[self.local_rank]

        self.outside_ddp = False  # 不允许在外部初始化
        self._data_device = kwargs.get('data_device', None)
        if isinstance(self._data_device, int):
            if self._data_device < 0:
                raise ValueError(
                    'Parameter `data_device` can not be smaller than 0.')
            _could_use_device_num = torch.cuda.device_count()
            if self._data_device >= _could_use_device_num:
                raise ValueError(
                    'The gpu device that parameter `device` specifies is '
                    'not existed.')
            self._data_device = torch.device(f'cuda:{self._data_device}')
        elif isinstance(self._data_device, str):
            self._data_device = torch.device(self._data_device)
        elif self._data_device is not None and not isinstance(
                self._data_device, torch.device):
            raise ValueError(
                'Parameter `device` is wrong type, please check our '
                'documentation for the right use.')

        self._master_port = None
        # world_size 表示的就是全局的显卡的数量；
        # int(os.environ.get("WORLD_SIZE"))  len(self.parallel_device)
        self.world_size = 0
        self.global_rank = 0

        if self.fs_type == 'ddp':
            if len(self.model._buffers) != 0 and self._ddp_kwargs.get(
                    'broadcast_buffers', None) is None:
                logger.info(
                    'Notice your model has buffers and you are using '
                    '`FairScaleDriver`, but you do not set '
                    "'broadcast_buffers' in your trainer. Cause in most "
                    "situations, this parameter can be set to 'False' to "
                    'avoid redundant data communication between different '
                    'processes.')

        self.output_from_new_proc = kwargs.get('output_from_new_proc',
                                               'only_error')
        assert isinstance(self.output_from_new_proc, str), \
            'Parameter `output_from_new_proc` can only be `str` type.'
        if self.output_from_new_proc not in {'all', 'ignore', 'only_error'}:
            os.makedirs(self.output_from_new_proc, exist_ok=True)
            self.output_from_new_proc = os.path.abspath(
                self.output_from_new_proc)

        # 设置这一参数是因为 evaluator 中也会进行 setup 操作，
        # 但是显然是不需要的也不应该的；
        self._has_setup = False
        # 判断传入的模型是否经过 _has_ddpwrapped 包裹；
        self._has_ddpwrapped = False

    def setup(self):
        r"""
        准备分布式环境，该函数主要做以下两件事情：

        1. 开启多进程，每个 gpu 设备对应单独的一个进程；
        2. 每个进程将模型迁移到自己对应的 ``gpu`` 设备上；然后使用
           ``DistributedDataParallel`` 包裹模型；
        """
        if self._has_setup:
            return
        self._has_setup = True
        if self.is_pull_by_torch_run:
            # dist.get_world_size() 只能在 dist.init_process_group 初始化
            # 之后进行调用；
            self.world_size = int(os.environ.get('WORLD_SIZE'))
            self.global_rank = int(os.environ.get('RANK'))
            logger.info(f'World size: {self.world_size}, '
                        f'Global rank: {self.global_rank}')

            if not dist.is_initialized():
                dist.init_process_group(
                    backend='nccl',
                    rank=self.global_rank,
                    world_size=self.world_size)

            os.environ['fastnlp_torch_launch_not_ddp'] = 'yes'
        else:
            if not dist.is_initialized():
                # 这里主要的问题在于要区分 rank0 和其它 rank 的情况；
                self.world_size = len(self.parallel_device)
                self.open_subprocess()
                # rank 一定是通过环境变量去获取的；
                self.global_rank = self.local_rank
                dist.init_process_group(
                    backend='nccl',
                    rank=self.global_rank,
                    world_size=self.world_size)
            # 用户在这个 trainer 前面又初始化了一个 trainer，并且
            # 使用的是 FairScaleDriver；
            else:
                # 如果 `dist.is_initialized() is True`，那么说明 FairScaleDriver
                # 在之前已经初始化并且已经 setup 过一次，那么我们需要保证现在使用的
                # （即之后的）FairScaleDriver 的设置和第一个 FairScaleDriver 是完
                # 全一样的；
                pre_num_processes = int(os.environ[FASTNLP_DISTRIBUTED_CHECK])
                if pre_num_processes != len(self.parallel_device):
                    raise RuntimeError(
                        'Notice you are using `FairScaleDriver` after one '
                        'instantiated `FairScaleDriver`, it is not allowed '
                        'that your second `FairScaleDriver` has a new setting '
                        'of parameters `num_nodes` and `num_processes`.')
                self.world_size = dist.get_world_size()
                self.global_rank = dist.get_rank()

        torch.cuda.set_device(self.model_device)
        if self.fs_type != 'fsdp':
            self.model.to(self.model_device)
        self.configure_ddp()

        self.barrier()
        # 初始化 self._pids，从而使得每一个进程都能接受到 rank0 的 send 操作；
        self._pids = [
            torch.tensor(0, dtype=torch.int).to(self.data_device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            self._pids,
            torch.tensor(os.getpid(), dtype=torch.int).to(self.data_device))
        local_world_size = int(os.environ.get(
            'LOCAL_WORLD_SIZE')) if 'LOCAL_WORLD_SIZE' in os.environ else None
        if local_world_size is None:
            local_world_size = torch.tensor(
                int(os.environ.get('LOCAL_RANK')),
                dtype=torch.int).to(self.data_device)
            dist.all_reduce(local_world_size, op=dist.ReduceOp.MAX)
            local_world_size = local_world_size.tolist() + 1

        node_rank = self.global_rank // local_world_size
        self._pids = self._pids[node_rank * local_world_size:(node_rank + 1) *
                                local_world_size]
        self._pids = self.tensor_to_numeric(self._pids)

    def configure_ddp(self):
        model = _DDPWrappingModel(self.model)
        if self.fs_type == 'ddp':
            self.model = DistributedDataParallel(
                # 注意这里的 self.model_device 是 `torch.device` type，
                # 因此 self.model_device.index；
                model,
                device_ids=[self.model_device.index],
                **self._ddp_kwargs)
        elif self.fs_type == 'sdp':
            sdp_kwargs = self._sdp_kwargs
            sdp_kwargs = {**sdp_kwargs, 'module': model}
            sdp_kwargs['reduce_fp16'] = sdp_kwargs.get('reduce_fp16',
                                                       self.fp16)
            oss_lst = []
            for optimizer in self.optimizers:
                oss = OSS(
                    optimizer.param_groups,
                    optim=type(optimizer),
                    **optimizer.defaults)
                oss_lst.append(oss)
            sdp_kwargs['sharded_optimizer'] = oss_lst
            sdp_kwargs['warn_on_trainable_params_changed'] = sdp_kwargs.get(
                'warn_on_trainable_params_changed', False)
            self.model = ShardedDataParallel(**sdp_kwargs)
            self.optimizers = oss_lst
        else:
            assert len(self.optimizers) == 1, "When fs_type='fsdp', only " \
                                              'one optimizer is allowed.'
            optimizer = self.optimizers[0]
            assert len(optimizer.param_groups) == 1, \
                'Cannot assign parameter specific optimizer parameter ' \
                "for 'fsdp'."
            fsdp_kwargs = self._fsdp_kwargs
            fsdp_kwargs['mixed_precision'] = self.fp16
            fsdp_kwargs['state_dict_on_rank_0_only'] = fsdp_kwargs.get(
                'state_dict_on_rank_0_only', True)
            fsdp_kwargs['state_dict_device'] = fsdp_kwargs.get(
                'state_dict_device', torch.device('cpu'))
            fsdp_kwargs['compute_device'] = fsdp_kwargs.get(
                'compute_device', self.model_device)
            optimizer = self.optimizers[0]
            # wrap_policy=functools.partial(
            #     default_auto_wrap_policy, min_num_params = 1e6)
            # with enable_wrap(wrapper_cls=FullyShardedDataParallel,
            #                  auto_wrap_policy=wrap_policy,
            #                  **fsdp_kwargs):
            #     model=auto_wrap(model)
            fsdp_kwargs = {**fsdp_kwargs, 'module': model}
            self.model = None  # 释放掉
            self.model = FullyShardedDataParallel(**fsdp_kwargs).to(
                self.model_device)
            self.optimizers = type(optimizer)(self.model.parameters(),
                                              **optimizer.defaults)

        self._has_ddpwrapped = True

    def save_model(  # type: ignore[override]
            self,
            filepath: Union[str, Path],
            only_state_dict: bool = True,
            **kwargs):
        """保存当前 driver 的模型到 folder 下。

        :param filepath: 保存到哪个文件夹；
        :param only_state_dict: 是否只保存权重；
        :return:
        """
        if self.fs_type in ('ddp', 'sdp'):
            model = self.model.module.model

        if only_state_dict:
            if self.fs_type != 'fsdp':
                if self.local_rank == 0:
                    states = {
                        name: param.cpu().detach().clone()
                        for name, param in model.state_dict().items()
                    }
            else:
                # 所有 rank 都需要调用
                states = self.model.state_dict()
                if self.local_rank == 0:
                    states = {
                        key[len('model.'):]: value
                        for key, value in states.items()
                    }  # 这里需要去掉那个 _wrap 的 key
            if self.local_rank == 0:  #
                torch.save(states, filepath)
        elif self.fs_type == 'fsdp':
            raise RuntimeError(
                "When fs_type='fsdp', only `only_state_dict=True` is allowed.")
        else:
            if self.local_rank == 0:
                torch.save(model, filepath)

    def load_model(  # type: ignore[override]
            self,
            filepath: str,
            only_state_dict: bool = True,
            **kwargs):
        """从 folder 中加载权重并赋值到当前 driver 的模型上。

        :param filepath: 加载权重或模型的路径
        :param load_state_dict: 保存的内容是否只是权重。
        :param kwargs:
        :return:
        """
        states = torch.load(filepath, map_location='cpu')
        if isinstance(states, dict) and only_state_dict is False:
            logger.rank_zero_warning(
                f'It seems like that {filepath} only contains state, '
                'you may need to use `only_state_dict=True`')
        elif not isinstance(states, dict) and only_state_dict is True:
            logger.rank_zero_warning(
                f'It seems like that {filepath} is not state, you may '
                'need to use `only_state_dict=False`')
        if not isinstance(states, Mapping):
            states = states.state_dict()

        if self.fs_type in ('ddp', 'sdp'):
            model = self.model.module.model
        else:
            model = self.model
            states = {f'model.{k}': v for k, v in states.items()}

        model.load_state_dict(states)

    def save_checkpoint(self,
                        folder: Path,
                        states: Dict,
                        dataloader,
                        only_state_dict: bool = True,
                        should_save_model: bool = True,
                        **kwargs):
        r"""
        断点重训的保存函数，该函数会负责保存 **优化器** 和 **sampler** 的状态，以及
        **模型** （若 ``should_save_model`` 为 ``True``）

        :param folder: 保存断点重训的状态的文件夹；:meth:`save_checkpoint` 函数应
            该在该路径下面下面新增名为 ``FASTNLP_CHECKPOINT_FILENAME`` 与
            ``FASTNLP_MODEL_FILENAME`` （若 ``should_save_model`` 为 ``True``）
            的文件。把 model 相关的内容放入到 ``FASTNLP_MODEL_FILENAME`` 文件中，
            将传入的 ``states`` 以及自身产生的其它状态一并保存在
            ``FASTNLP_CHECKPOINT_FILENAME`` 里面。
        :param states: 由 :class:`.Trainer` 传入的一个字典，其中已经包含了为了实现
            断点重训所需要保存的其它对象的状态。
        :param dataloader: 正在使用的 dataloader。
        :param only_state_dict: 是否只保存模型的参数，当 ``should_save_model``
            为 ``False``，该参数无效。
        :param should_save_model: 是否应该保存模型，如果为 ``False``，Driver 将不
            负责 model 的保存。
        """
        if self.fs_type == 'fsdp':
            if should_save_model is False:
                logger.warning(
                    "When save model using fs_type='fsdp', please make sure "
                    'use `with trainer.driver.model.summon_full_params():` '
                    'context to gather all parameters.')
            with all_rank_call_context():
                super().save_checkpoint(
                    folder=folder,
                    states=states,
                    dataloader=dataloader,
                    only_state_dict=only_state_dict,
                    should_save_model=should_save_model,
                    **kwargs)
        else:
            super().save_checkpoint(
                folder=folder,
                states=states,
                dataloader=dataloader,
                only_state_dict=only_state_dict,
                should_save_model=should_save_model,
                **kwargs)

    def get_optimizer_state(self):
        optimizers_state_dict = {}
        for i in range(len(self.optimizers)):
            optimizer: torch.optim.Optimizer = self.optimizers[i]
            if self.fs_type == 'fsdp':
                optimizer_state = self.model.gather_full_optim_state_dict(
                    optimizer)
            elif self.fs_type == 'sdp':
                optimizer.consolidate_state_dict(recipient_rank=0)
                optimizer_state = optimizer.state_dict()
            else:
                optimizer_state = optimizer.state_dict()
            if self.local_rank == 0:
                optimizer_state['state'] = optimizer_state_to_device(
                    optimizer_state['state'], torch.device('cpu'))
                # 注意这里没有使用 deepcopy，测试是不需要的；
                optimizers_state_dict[f'optimizer{i}'] = optimizer_state
        return optimizers_state_dict

    def load_optimizer_state(self, states):
        assert len(states) == len(self.optimizers), \
            f'The number of optimizers is:{len(self.optimizers)}, while in ' \
            f'checkpoint it is:{len(states)}'
        for i in range(len(self.optimizers)):
            optimizer: torch.optim.Optimizer = self.optimizers[i]
            state = states[f'optimizer{i}']
            if self.fs_type == 'fsdp':
                state = self.model.get_shard_from_optim_state_dict(state)
            optimizer.load_state_dict(state)

        logger.debug('Load optimizer state dict.')

    def unwrap_model(self):
        r"""
        :return: 原本的模型，例如没有被 ``DataParallel`` 包裹；
        """
        return self.model.module.model

__all__ = [
    'FairScaleDriver'
]
from typing import List, Sequence, Union, Dict, Mapping
from pathlib import Path
import os
import functools

from fastNLP.envs.imports import _NEED_IMPORT_FAIRSCALE
if _NEED_IMPORT_FAIRSCALE:
    import torch
    import torch.distributed as dist
    from fairscale.optim import OSS
    from fairscale.nn import ShardedDataParallel
    from fairscale.nn import FullyShardedDataParallel
    from fairscale.optim.grad_scaler import ShardedGradScaler
    from torch.nn.parallel import DistributedDataParallel
    from fairscale.nn.wrap import auto_wrap, enable_wrap, default_auto_wrap_policy

from ...log import logger
from .utils import _DDPWrappingModel

from .ddp import TorchDDPDriver
from .torch_driver import TorchDriver
from .utils import _build_fp16_env
from ....envs.distributed import all_rank_call_context
from fastNLP.envs import FASTNLP_DISTRIBUTED_CHECK
from .utils import optimizer_state_to_device


class FairScaleDriver(TorchDDPDriver):
    def __init__(
            self,
            model,
            parallel_device: Union[List["torch.device"], "torch.device"],
            is_pull_by_torch_run = False,
            fp16: bool = False,
            fairscale_kwargs: Dict = None,
            **kwargs
    ):
        assert _NEED_IMPORT_FAIRSCALE, "fairscale is not imported."
        assert not dist.is_initialized(), "FairScaleDriver does not support initialize distributed by user."
        self._fairscale_kwargs = fairscale_kwargs
        self.fs_type = self._fairscale_kwargs.get('fs_type', 'sdp')  # ddp, sdp, fsdp
        if self.fs_type == 'fsdp':
            self._fairscale_kwargs['set_grad_to_none'] = self._fairscale_kwargs.get('set_grad_to_none', True)
        # 将最顶上的进行初始化
        kwargs.pop('torch_kwargs', None)
        TorchDriver.__init__(self, model=model, fp16=False, torch_kwargs=self._fairscale_kwargs, **kwargs)
        self.is_pull_by_torch_run = is_pull_by_torch_run
        assert self.fs_type in ['ddp', 'sdp', 'fsdp']
        self._oss_kwargs = self._fairscale_kwargs.get('oss_kwargs', {})  # 仅在 ddp 和 sdp 下有使用到
        self._sdp_kwargs = self._fairscale_kwargs.get('sdp_kwargs', {})
        self._fdsp_kwargs = self._fairscale_kwargs.get('fsdp_kwargs', {})
        self._ddp_kwargs = self._fairscale_kwargs.get('ddp_kwargs', {})

        if self.fs_type == 'ddp' or fp16 is False:
            self.auto_cast, _grad_scaler = _build_fp16_env(dummy=not fp16)
            self.grad_scaler = _grad_scaler(**self._fairscale_kwargs.get('gradscaler_kwargs', {}))
        else:
            self.auto_cast, self.grad_scaler = torch.cuda.amp.autocast, \
                                                   ShardedGradScaler(**self._fairscale_kwargs.get('gradscaler_kwargs', {}))

        self.parallel_device = parallel_device
        if is_pull_by_torch_run:
            self.model_device = parallel_device
        else:
            self.model_device = parallel_device[self.local_rank]

        self.outside_ddp = False  # 不允许在外部初始化
        self._data_device = kwargs.get("data_device", None)
        if isinstance(self._data_device, int):
            if self._data_device < 0:
                raise ValueError("Parameter `data_device` can not be smaller than 0.")
            _could_use_device_num = torch.cuda.device_count()
            if self._data_device >= _could_use_device_num:
                raise ValueError("The gpu device that parameter `device` specifies is not existed.")
            self._data_device = torch.device(f"cuda:{self._data_device}")
        elif isinstance(self._data_device, str):
            self._data_device = torch.device(self._data_device)
        elif self._data_device is not None and not isinstance(self._data_device, torch.device):
            raise ValueError("Parameter `device` is wrong type, please check our documentation for the right use.")

        self._master_port = None
        # world_size 表示的就是全局的显卡的数量；
        self.world_size = None  # int(os.environ.get("WORLD_SIZE"))  len(self.parallel_device)
        self.global_rank = 0

        if self.fs_type == 'ddp':
            if len(self.model._buffers) != 0 and self._ddp_kwargs.get("broadcast_buffers", None) is None:
                logger.info("Notice your model has buffers and you are using `FairScaleDriver`, but you do not set "
                            "'broadcast_buffers' in your trainer. Cause in most situations, this parameter can be set"
                            " to 'False' to avoid redundant data communication between different processes.")

        self.output_from_new_proc = kwargs.get("output_from_new_proc", "only_error")
        assert isinstance(self.output_from_new_proc, str), "Parameter `output_from_new_proc` can only be `str` type."
        if self.output_from_new_proc not in {"all", "ignore", "only_error"}:
            os.makedirs(self.output_from_new_proc, exist_ok=True)
            self.output_from_new_proc = os.path.abspath(self.output_from_new_proc)

        self._has_setup = False  # 设置这一参数是因为 evaluator 中也会进行 setup 操作，但是显然是不需要的也不应该的；
        self._has_ddpwrapped = False  # 判断传入的模型是否经过 _has_ddpwrapped 包裹；

    def setup(self):
        r"""
        准备分布式环境，该函数主要做以下两件事情：

            1. 开启多进程，每个 gpu 设备对应单独的一个进程；
            2. 每个进程将模型迁移到自己对应的 ``gpu`` 设备上；然后使用 ``DistributedDataParallel`` 包裹模型；
        """
        if self._has_setup:
            return
        self._has_setup = True
        if self.is_pull_by_torch_run:
            # dist.get_world_size() 只能在 dist.init_process_group 初始化之后进行调用；
            self.world_size = int(os.environ.get("WORLD_SIZE"))
            self.global_rank = int(os.environ.get("RANK"))
            logger.info(f"World size: {self.world_size}, Global rank: {self.global_rank}")

            if not dist.is_initialized():
                dist.init_process_group(
                    backend="nccl", rank=self.global_rank, world_size=self.world_size
                )

            os.environ["fastnlp_torch_launch_not_ddp"] = "yes"
        else:
            if not dist.is_initialized():
                # 这里主要的问题在于要区分 rank0 和其它 rank 的情况；
                self.world_size = len(self.parallel_device)
                self.open_subprocess()
                self.global_rank = self.local_rank  # rank 一定是通过环境变量去获取的；
                dist.init_process_group(
                    backend="nccl", rank=self.global_rank, world_size=self.world_size
                )
            # 用户在这个 trainer 前面又初始化了一个 trainer，并且使用的是 TorchDDPDriver；
            else:
                # 如果 `dist.is_initialized() == True`，那么说明 TorchDDPDriver 在之前已经初始化并且已经 setup 过一次，那么我们需要保证现在
                #  使用的（即之后的）TorchDDPDriver 的设置和第一个 TorchDDPDriver 是完全一样的；
                pre_num_processes = int(os.environ[FASTNLP_DISTRIBUTED_CHECK])
                if pre_num_processes != len(self.parallel_device):
                    raise RuntimeError(
                        "Notice you are using `TorchDDPDriver` after one instantiated `TorchDDPDriver`, it is not"
                        "allowed that your second `TorchDDPDriver` has a new setting of parameters "
                        "`num_nodes` and `num_processes`.")
                self.world_size = dist.get_world_size()
                self.global_rank = dist.get_rank()

        torch.cuda.set_device(self.model_device)
        if self.fs_type != 'fsdp':
            self.model.to(self.model_device)
        self.configure_ddp()

        self.barrier()
        # 初始化 self._pids，从而使得每一个进程都能接受到 rank0 的 send 操作；
        self._pids = [torch.tensor(0, dtype=torch.int).to(self.data_device) for _ in range(dist.get_world_size())]
        dist.all_gather(self._pids, torch.tensor(os.getpid(), dtype=torch.int).to(self.data_device))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE")) if "LOCAL_WORLD_SIZE" in os.environ else None
        if local_world_size is None:
            local_world_size = torch.tensor(int(os.environ.get("LOCAL_RANK")), dtype=torch.int).to(self.data_device)
            dist.all_reduce(local_world_size, op=dist.ReduceOp.MAX)
            local_world_size = local_world_size.tolist() + 1

        node_rank = self.global_rank // local_world_size
        self._pids = self._pids[node_rank * local_world_size: (node_rank + 1) * local_world_size]
        self._pids = self.tensor_to_numeric(self._pids)

    def configure_ddp(self):
        model = _DDPWrappingModel(self.model)
        if self.fs_type == 'ddp':
            self.model = DistributedDataParallel(
                # 注意这里的 self.model_device 是 `torch.device` type，因此 self.model_device.index；
                model, device_ids=[self.model_device.index],
                **self._ddp_kwargs
            )
        elif self.fs_type == 'sdp':
            sdp_kwargs = self._sdp_kwargs
            sdp_kwargs = {**sdp_kwargs, 'module': model}
            sdp_kwargs['reduce_fp16'] = sdp_kwargs.get('reduce_fp16', self.fp16)
            oss_lst = []
            for optimizer in self.optimizers:
                oss = OSS(optimizer.param_groups, optim=type(optimizer), **optimizer.defaults)
                oss_lst.append(oss)
            sdp_kwargs['sharded_optimizer'] = oss_lst
            sdp_kwargs['warn_on_trainable_params_changed'] = sdp_kwargs.get('warn_on_trainable_params_changed', False)
            self.model = ShardedDataParallel(**sdp_kwargs)
            self.optimizers = oss_lst
        else:
            assert len(self.optimizers) == 1, "When fs_type='fsdp', only one optimizer is allowed."
            optimizer = self.optimizers[0]
            assert len(optimizer.param_groups) == 1, "Cannot assign parameter specific optimizer parameter for 'fsdp'."
            fsdp_kwargs = self._fdsp_kwargs
            fsdp_kwargs['mixed_precision'] = self.fp16
            fsdp_kwargs['state_dict_on_rank_0_only'] = fsdp_kwargs.get('state_dict_on_rank_0_only', True)
            fsdp_kwargs['state_dict_device'] = fsdp_kwargs.get('state_dict_device', torch.device('cpu'))
            fsdp_kwargs['compute_device'] = fsdp_kwargs.get('compute_device', self.model_device)
            optimizer = self.optimizers[0]
            # wrap_policy = functools.partial(default_auto_wrap_policy, min_num_params=1e6)
            # with enable_wrap(wrapper_cls=FullyShardedDataParallel, auto_wrap_policy=wrap_policy,
            #                  **fsdp_kwargs):
            #     model = auto_wrap(model)
            fsdp_kwargs = {**fsdp_kwargs, 'module': model}
            self.model = None  # 释放掉
            self.model = FullyShardedDataParallel(**fsdp_kwargs).to(self.model_device)
            self.optimizers = type(optimizer)(self.model.parameters(), **optimizer.defaults)

        self._has_ddpwrapped = True

    def save_model(self, filepath: Union[str, Path], only_state_dict: bool = True, **kwargs):
        """
        保存当前 driver 的模型到 folder 下。

        :param filepath: 保存到哪个文件夹；
        :param only_state_dict: 是否只保存权重；
        :return:
        """
        if self.fs_type in ('ddp', 'sdp'):
            model = self.model.module.model

        if only_state_dict:
            if self.fs_type != 'fsdp':
                if self.local_rank == 0:
                    states = {name: param.cpu().detach().clone() for name, param in model.state_dict().items()}
            else:
                # 所有 rank 都需要调用
                states = self.model.state_dict()
                if self.local_rank == 0:
                    states = {key[len('model.'):]:value for key, value in states.items()}  # 这里需要去掉那个 _wrap 的 key
            if self.local_rank == 0:  #
                torch.save(states, filepath)
        elif self.fs_type == 'fsdp':
            raise RuntimeError("When fs_type='fsdp', only `only_state_dict=True` is allowed.")
        else:
            if self.local_rank == 0:
                torch.save(model, filepath)

    def load_model(self, filepath: str, only_state_dict: bool = True, **kwargs):
        """
        从 folder 中加载权重并赋值到当前 driver 的模型上。

        :param filepath: 加载权重或模型的路径
        :param load_state_dict: 保存的内容是否只是权重。
        :param kwargs:
        :return:
        """
        states = torch.load(filepath, map_location='cpu')
        if isinstance(states, dict) and only_state_dict is False:
            logger.rank_zero_warning(f"It seems like that {filepath} only contains state, you may need to use "
                                     f"`only_state_dict=True`")
        elif not isinstance(states, dict) and only_state_dict is True:
            logger.rank_zero_warning(f"It seems like that {filepath} is not state, you may need to use "
                                     f"`only_state_dict=False`")
        if not isinstance(states, Mapping):
            states = states.state_dict()

        if self.fs_type in ('ddp', 'sdp'):
            model = self.model.module.model
        else:
            model = self.model
            states = {f'model.{k}':v for k, v in states.items()}

        model.load_state_dict(states)

    def save_checkpoint(self, folder: Path, states: Dict, dataloader, only_state_dict: bool = True, should_save_model: bool = True, **kwargs):
        if self.fs_type == 'fsdp':
            if should_save_model is False:
                logger.warning("When save model using fs_type='fsdp', please make sure use "
                               "`with trainer.driver.model.summon_full_params():` context to gather all parameters.")
            with all_rank_call_context():
                super().save_checkpoint(folder=folder, states=states, dataloader=dataloader, only_state_dict=only_state_dict,
                                        should_save_model=should_save_model, **kwargs)
        else:
            super().save_checkpoint(folder=folder, states=states, dataloader=dataloader,
                                    only_state_dict=only_state_dict, should_save_model=should_save_model, **kwargs)

    def get_optimizer_state(self):
        optimizers_state_dict = {}
        for i in range(len(self.optimizers)):
            optimizer: torch.optim.Optimizer = self.optimizers[i]
            if self.fs_type == 'fsdp':
                optimizer_state = self.model.gather_full_optim_state_dict(optimizer)
            elif self.fs_type == 'sdp':
                optimizer.consolidate_state_dict(recipient_rank=0)
            else:
                optimizer_state = optimizer.state_dict()
            if self.local_rank == 0:
                optimizer_state["state"] = optimizer_state_to_device(optimizer_state["state"], torch.device("cpu"))
                optimizers_state_dict[f"optimizer{i}"] = optimizer_state  # 注意这里没有使用 deepcopy，测试是不需要的；
        return optimizers_state_dict

    def load_optimizer_state(self, states):
        assert len(states) == len(self.optimizers), f"The number of optimizers is:{len(self.optimizers)}, while in " \
                                                    f"checkpoint it is:{len(states)}"
        for i in range(len(self.optimizers)):
            optimizer: torch.optim.Optimizer = self.optimizers[i]
            state = states[f'optimizer{i}']
            if self.fs_type == 'fsdp':
                state = self.model.get_shard_from_optim_state_dict(state)
            optimizer.load_state_dict(state)

        logger.debug("Load optimizer state dict.")

    def unwrap_model(self):
        r"""
        :return: 原本的模型，例如没有被 ``DataParallel`` 包裹；
        """
        return self.model.module.model

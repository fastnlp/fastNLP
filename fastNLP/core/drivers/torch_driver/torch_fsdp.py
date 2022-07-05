


from fastNLP.envs.imports import _TORCH_GREATER_EQUAL_1_12

if _TORCH_GREATER_EQUAL_1_12:
    from torch.distributed.fsdp import FullyShardedDataParallel, StateDictType, FullStateDictConfig, OptimStateKeyType

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from typing import Optional, Union, List, Dict, Mapping
from pathlib import Path

from .ddp import TorchDDPDriver
from fastNLP.core.drivers.torch_driver.utils import (
    _DDPWrappingModel,
)

from fastNLP.envs import FASTNLP_DISTRIBUTED_CHECK, FASTNLP_MODEL_FILENAME, FASTNLP_CHECKPOINT_FILENAME, \
    FASTNLP_GLOBAL_RANK, rank_zero_call
from fastNLP.core.drivers.torch_driver.utils import DummyGradScaler
from fastNLP.core.log import logger
from fastNLP.core.utils import check_user_specific_params
from .utils import optimizer_state_to_device


"""
参考文档：
1. https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/
2. https://pytorch.org/docs/stable/fsdp.html?highlight=fsdp
3. https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
4. https://engineering.fb.com/2021/07/15/open-source/fsdp/
"""

class TorchFSDPDriver(TorchDDPDriver):
    r"""
    实现对于 pytorch 自己实现的 fully sharded data parallel；请阅读该文档了解更多：
    https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict；

    ..note::

        ``TorchFSDPDriver`` 大部分行为与 ``TorchDDPDriver`` 相同，如果您不了解 ``TorchDDPDriver``，
        您可以先阅读 :class:`~fastNLP.core.drivers.TorchDDPDriver`；

    ..warning::

        ``TorchFSDPDriver`` 现在还不支持断点重训功能，但是支持保存模型和加载模型；

        注意当您在加载和保存模型的 checkpointcallback 的时候，您可以通过在初始化 ``Trainer`` 时传入
        ``torch_kwargs={"fsdp_kwargs": {'save_on_rank0': True/False, 'load_on_rank0': True/False}}`` 来指定保存模型的行为：

            1. save/load_on_rank0 = True：表示在加载和保存模型时将所有 rank 上的模型参数全部聚合到 rank0 上，注意这样可能会造成 OOM；
            2. save/load_on_rank0 = False：表示每个 rank 分别保存加载自己独有的模型参数；

    """

    def __init__(
            self,
            model,
            parallel_device: Optional[Union[List["torch.device"], "torch.device"]],
            is_pull_by_torch_run: bool = False,
            fp16: bool = False,
            torch_kwargs: Dict = None,
            **kwargs
    ):

        # 在加入很多东西后，需要注意这里调用 super 函数的位置；
        super(TorchDDPDriver, self).__init__(model, fp16=fp16, torch_kwargs=torch_kwargs, **kwargs)

        if isinstance(model, torch.nn.DataParallel):
            raise ValueError(f"Parameter `model` can not be `DataParallel` in `TorchDDPDriver`, it should be "
                             f"`torch.nn.Module` or `torch.nn.parallel.DistributedDataParallel` type.")

        # 如果用户自己在外面初始化 DDP，那么其一定是通过 python -m torch.distributed.launch 拉起的；
        self.is_pull_by_torch_run = is_pull_by_torch_run
        self.parallel_device = parallel_device
        if not is_pull_by_torch_run and parallel_device is None:
            raise ValueError(
                "Parameter `parallel_device` can not be None when using `TorchDDPDriver`. This error is caused "
                "when your value of parameter `device` is `None` in your `Trainer` instance.")

        # 注意我们在 initialize_torch_driver 中的逻辑就是如果是 is_pull_by_torch_run，那么我们就直接把 parallel_device 置为当前进程的gpu；
        if is_pull_by_torch_run:
            self.model_device = parallel_device
        else:
            # 我们的 model_device 一定是 torch.device，而不是一个 list；
            self.model_device = parallel_device[self.local_rank]

        # 如果用户自己在外面初始化了 FSDP；
        self.outside_ddp = False
        if dist.is_initialized() and FASTNLP_DISTRIBUTED_CHECK not in os.environ and \
                "fastnlp_torch_launch_not_ddp" not in os.environ:
            # 如果用户自己在外面初始化了 DDP，那么我们要求用户传入的模型一定是已经由 DistributedDataParallel 包裹后的模型；
            if not isinstance(model, FullyShardedDataParallel):
                raise RuntimeError(
                    "It is not allowed to input a normal model instead of `FullyShardedDataParallel` when"
                    "you initialize the ddp process out of our control.")
            if isinstance(model, DistributedDataParallel):
                logger.warning("You are using `TorchFSDPDriver`, but you have initialized your model as "
                               "`DistributedDataParallel`, which will make the `FullyShardedDataParallel` not work "
                               "as expected. You could just delete `DistributedDataParallel` wrap operation.")

            self.outside_ddp = True
            # 用户只有将模型上传到对应机器上后才能用 DistributedDataParallel 包裹，因此如果用户在外面初始化了 DDP，那么在 TorchDDPDriver 中
            #  我们就直接将 model_device 置为 None；
            self.model_device = None

        # 当用户自己在外面初始化 DDP 时我们会将 model_device 置为 None，这是用户可以通过 `data_device` 将对应的数据移到指定的机器上;
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

        self._fsdp_kwargs = self._torch_kwargs.get("fsdp_kwargs", {})
        self._save_on_rank0 = self._fsdp_kwargs.get("save_on_rank0", False)
        if "save_on_rank0" in self._fsdp_kwargs:
            self._fsdp_kwargs.pop("save_on_rank0")
        self._load_on_rank0 = self._fsdp_kwargs.get("load_on_rank0", False)
        if "load_on_rank0" in self._fsdp_kwargs:
            self._fsdp_kwargs.pop("load_on_rank0")

        if self._save_on_rank0 != self._load_on_rank0:
            logger.warning(f"Notice the behavior between ``save`` and ``load`` is not matched, you choose "
                           f"{'save on rank0' if self._save_on_rank0 else 'save on each rank'}, but "
                           f"{'load on rank0' if self._save_on_rank0 else 'load on each rank'}!")

        check_user_specific_params(self._fsdp_kwargs, FullyShardedDataParallel.__init__, FullyShardedDataParallel.__name__)
        if "cpu_offload" in self._fsdp_kwargs and kwargs["accumulation_steps"] != 1:
            logger.warning("It is not supported ``accumulation_steps`` when using ``cpu_offload`` in "
                           "``FullyShardedDataParallel``.")

        self.output_from_new_proc = kwargs.get("output_from_new_proc", "only_error")
        assert isinstance(self.output_from_new_proc, str), "Parameter `output_from_new_proc` can only be `str` type."
        if self.output_from_new_proc not in {"all", "ignore", "only_error"}:
            os.makedirs(name=self.output_from_new_proc, exist_ok=True)
            self.output_from_new_proc = os.path.abspath(self.output_from_new_proc)

        self._has_setup = False  # 设置这一参数是因为 evaluator 中也会进行 setup 操作，但是显然是不需要的也不应该的；
        self._has_ddpwrapped = False  # 判断传入的模型是否经过 _has_ddpwrapped 包裹；

    def configure_ddp(self):
        torch.cuda.set_device(self.model_device)
        if not isinstance(self.model, FullyShardedDataParallel):
            self.model = FullyShardedDataParallel(
                # 注意这里的 self.model_device 是 `torch.device` type，因此 self.model_device.index；
                _DDPWrappingModel(self.model), device_id=self.model_device.index,
                **self._fsdp_kwargs
            )

            # 必须先使用 FullyShardedDataParallel 包裹模型后再使用 optimizer 包裹模型的参数，因此这里需要将 optimizer 重新初始化一遍；
            for i in range(len(self.optimizers)):
                self.optimizers[i] = type(self.optimizers[i])(self.model.parameters(), **self.optimizers[i].defaults)

            self._has_ddpwrapped = True

    def unwrap_model(self):
        """
        注意该函数因为需要在特定的时候进行调用，例如 ddp 在 get_model_call_fn 的时候，因此不能够删除；
        如果您使用该函数来获取原模型的结构信息，是可以的；
        但是如果您想要通过该函数来获取原模型实际的参数，是不可以的，因为在 FullyShardedDataParallel 中模型被切分成了多个部分，而对于每个 gpu 上
        的模型只是整体模型的一部分。
        """
        _module = self.model.module.module
        if isinstance(_module, _DDPWrappingModel):
            return _module.model
        else:
            return _module

    def save_model(self, filepath: Union[str, Path], only_state_dict: bool = True, **kwargs):
        filepath = Path(filepath)
        prefix = filepath.parent
        filename = filepath.name
        _filename = filename.split('.')
        filename, suffix = _filename[0], '.'.join(_filename[1:])
        if only_state_dict:
            if self._save_on_rank0:
                full_state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FullyShardedDataParallel.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, full_state_dict_config):
                    state_dict = self.model.state_dict()
                rank_zero_call(torch.save)(state_dict, filepath)
            else:
                # 添加 'rank0/1' 字段来区分全部聚集到 rank0 保存的方式；
                _filename = filename.split('_')
                filename = _filename[0] + f"_rank{int(os.environ.get(FASTNLP_GLOBAL_RANK, 0))}_" + _filename[1]
                filepath = prefix.joinpath(filename + "." + suffix)
                with FullyShardedDataParallel.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
                    state_dict = self.model.state_dict()
                torch.save(state_dict, filepath)
        else:
            raise RuntimeError("When using `TorchFSDPDriver`, only `only_state_dict=True` is allowed.")

    def load_model(self, filepath: Union[Path, str], only_state_dict: bool = True, **kwargs):
        if only_state_dict is False:
            raise RuntimeError("When using `TorchFSDPDriver`, only `only_state_dict=True` is allowed.")
        filepath = Path(filepath)
        prefix = filepath.parent
        filename = filepath.name
        _filename = filename.split('.')
        filename, suffix = _filename[0], '.'.join(_filename[1:])

        if not self._load_on_rank0:
            _filename = filename.split('_')
            filename = _filename[0] + f"_rank{int(os.environ.get(FASTNLP_GLOBAL_RANK, 0))}_" + _filename[1]
            filepath = prefix.joinpath(filename + "." + suffix)
            states = torch.load(filepath)
        else:
            states = torch.load(filepath, map_location="cpu")

        if isinstance(states, dict) and only_state_dict is False:
            logger.rank_zero_warning(f"It seems like that {filepath} only contains state, you may need to use "
                                     f"`only_state_dict=True`")
        elif not isinstance(states, dict) and only_state_dict is True:
            logger.rank_zero_warning(f"It seems like that {filepath} is not state, you may need to use "
                                     f"`only_state_dict=False`")
        if not isinstance(states, Mapping):
            states = states.state_dict()

        if self._load_on_rank0:
            with FullyShardedDataParallel.state_dict_type(self.model, StateDictType.FULL_STATE_DICT):
                self.model.load_state_dict(states)
        else:
            with FullyShardedDataParallel.state_dict_type(self.model, StateDictType.LOCAL_STATE_DICT):
                self.model.load_state_dict(states)

    def save_checkpoint(self, folder: Path, states: Dict, dataloader, only_state_dict: bool = True, should_save_model: bool = True, **kwargs):
        raise RuntimeError("``TorchFSDPDriver`` does not support ``save_checkpoint`` function for now, there is some "
                           "technical issues that needs to solve. You can implement your own breakpoint retraining "
                           "by rewriting this function. The important thing is how to save and load the optimizers' state dict, "
                           "you can see ``https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict``.")

    def load_checkpoint(self, folder: Path, dataloader, only_state_dict: bool = True, should_load_model: bool = True, **kwargs) -> Dict:
        raise RuntimeError("``TorchFSDPDriver`` does not support ``load_checkpoint`` function for now, there is some "
                           "technical issues that needs to solve. You can implement your own breakpoint retraining "
                           "by rewriting this function. The important thing is how to save and load the optimizers' state dict, "
                           "you can see ``https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.full_optim_state_dict``.")

    # todo 这些加了 __ 的函数是目前还不支持；
    #  这是因为 1.12 的 pytorch fsdp 的关于如何保存和加载 optimizer state dict 的接口有点过于反人类，无法在 fastNLP 的框架中进行调和
    #  使用；
    def __get_optimizer_state(self):
        optimizers_state_dict = {}
        for i in range(len(self.optimizers)):
            # 注意这里其余 rank 拿到的是一个空字典，因此在真正保存的时候需要保证只有 rank0 在工作；
            optimizer_state = FullyShardedDataParallel.full_optim_state_dict(self.model, self.optimizers[i])
            if self._save_on_rank0:
                with FullyShardedDataParallel.summon_full_params(self.model):
                    if int(os.environ.get(FASTNLP_GLOBAL_RANK, 0)) == 0:
                        unwrapped_model = self.model.module.module
                        optimizer_state = FullyShardedDataParallel.rekey_optim_state_dict(
                            optimizer_state, OptimStateKeyType.PARAM_ID, unwrapped_model)
                if int(os.environ.get(FASTNLP_GLOBAL_RANK, 0)) == 0:
                    optimizer_state["state"] = optimizer_state_to_device(optimizer_state["state"], torch.device("cpu"))
            optimizers_state_dict[f"optimizer{i}"] = optimizer_state  # 注意这里没有使用 deepcopy，测试是不需要的；
        return optimizers_state_dict

    # 这里单独拿出来是因为对于 fsdp 来说，每一个进程都需要运行此函数，因此不能包裹 rank_zero_call；
    def __save_checkpoint(self, folder: Path, states: Dict, dataloader, only_state_dict: bool = True, should_save_model: bool = True, **kwargs):
        if not only_state_dict:
            raise RuntimeError("When using `TorchFSDPDriver`, only `only_state_dict=True` is allowed.")

        # 1. sampler 的状态；
        num_consumed_batches = states.pop('num_consumed_batches')
        states['sampler_states'] = self.get_sampler_state(dataloader, num_consumed_batches)

        # 2. 保存模型的状态；
        if should_save_model:
            if not os.path.exists(folder):
                os.mkdir(folder)
            model_path = folder.joinpath(FASTNLP_MODEL_FILENAME)
            self.save_model(model_path, only_state_dict=True)

        # 3. 保存 optimizers 的状态；
        states["optimizers_state_dict"] = self.get_optimizer_state()
        logger.debug("Save optimizer state dict.")

        # 4. 保存fp16的状态
        if not isinstance(self.grad_scaler, DummyGradScaler):
            grad_scaler_state_dict = self.grad_scaler.state_dict()
            states['grad_scaler_state_dict'] = grad_scaler_state_dict

        # 确保只有 rank0 才会执行实际的保存操作；
        rank_zero_call(torch.save)(states, Path(folder).joinpath(FASTNLP_CHECKPOINT_FILENAME))

    def __load_optimizer_state(self, states):
        assert len(states) == len(self.optimizers), f"The number of optimizers is:{len(self.optimizers)}, while in " \
                                                    f"checkpoint it is:{len(states)}"

        with FullyShardedDataParallel.summon_full_params(self.model):
            unwrapped_model = self.model.module.module

            for i in range(len(self.optimizers)):
                optimizer_state = states[f'optimizer{i}']
                if self._load_on_rank0:
                    optimizer_state = FullyShardedDataParallel.rekey_optim_state_dict(optimizer_state, OptimStateKeyType.PARAM_NAME, unwrapped_model)
                optimizer_state = FullyShardedDataParallel.shard_full_optim_state_dict(optimizer_state, unwrapped_model)
                optimizer: torch.optim.Optimizer = type(self.optimizers[i])(unwrapped_model.parameters(), **self.optimizers[i].defaults)
                optimizer.load_state_dict(optimizer_state)
                self.optimizers[i] = optimizer

        logger.debug("Load optimizer state dict.")

    def __load_checkpoint(self, folder: Path, dataloader, only_state_dict: bool = True, should_load_model: bool = True, **kwargs) -> Dict:
        if not only_state_dict:
            raise RuntimeError("When using `TorchFSDPDriver`, only `only_state_dict=True` is allowed.")

        states = torch.load(folder.joinpath(FASTNLP_CHECKPOINT_FILENAME))

        # 1. 加载 optimizers 的状态；
        optimizers_state_dict = states.pop("optimizers_state_dict")
        self.load_optimizer_state(optimizers_state_dict)

        # 2. 加载模型状态；
        if should_load_model:
            self.load_model(filepath=folder.joinpath(FASTNLP_MODEL_FILENAME), only_state_dict=only_state_dict)

        # 3. 加载 fp16 的状态
        if "grad_scaler_state_dict" in states:
            grad_scaler_state_dict = states.pop("grad_scaler_state_dict")
            if not isinstance(self.grad_scaler, DummyGradScaler):
                self.grad_scaler.load_state_dict(grad_scaler_state_dict)
                logger.debug("Load grad_scaler state dict...")
        elif not isinstance(self.grad_scaler, DummyGradScaler):
            logger.rank_zero_warning(f"Checkpoint {folder} is not trained with fp16=True, while resume to a fp16=True training, "
                           f"the training process may be unstable.")

        # 4. 恢复 sampler 的状态；
        sampler_states = states.pop('sampler_states')
        states_ret = self.load_sampler_state(dataloader, sampler_states)
        states.update(states_ret)

        return states


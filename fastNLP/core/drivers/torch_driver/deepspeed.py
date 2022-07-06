import os
import argparse
import logging
from pathlib import Path

from typing import Union, Dict, List
from .torch_driver import TorchDriver
from .ddp import TorchDDPDriver
from .utils import _create_default_config, _DeepSpeedWrappingModel
from fastNLP.core.utils import nullcontext
from fastNLP.core.log import logger
from fastNLP.envs import(
    FASTNLP_DISTRIBUTED_CHECK,
    FASTNLP_CHECKPOINT_FILENAME
)
from fastNLP.envs.imports import _NEED_IMPORT_TORCH, _NEED_IMPORT_DEEPSPEED

if _NEED_IMPORT_TORCH:
    import torch
    import torch.distributed as dist
    from torch.optim import Optimizer
    
if _NEED_IMPORT_DEEPSPEED:
    import deepspeed
    from deepspeed import DeepSpeedEngine, DeepSpeedOptimizer

__all__ = [
    "DeepSpeedDriver",
]

class DeepSpeedDriver(TorchDDPDriver):
    """
    实现 ``deepspeed`` 分布式训练的 ``Driver``。

    .. note::

        您在绝大多数情况下不需要自己使用到该类，通过向 ``Trainer`` 传入正确的参数，您可以方便快速地部署您的分布式训练；

        ``DeepSpeedDriver`` 目前支持的三种启动方式：

            1. 用户自己不进行任何操作，直接使用我们的 ``Trainer``，这时是由我们自己使用 ``open_subprocesses`` 拉起多个进程，
            然后 ``DeepSpeedDriver`` 自己通过调用 ``deepspeed.initialize`` 来初始化模型和同心组；（情况 A）

            .. code-block::

                trainer = Trainer(
                    ...
                    driver='deepspeed',
                    device=[0, 1]
                )
                trainer.run()

            通过运行 ``python train.py`` 启动；

            2. 用户同样不在 ``Trainer`` 之外初始化 ``deepspeed``，但是用户自己使用 ``python -m torch.distributed.launch`` 拉起来创建多个进程，这时我们仍旧
            会通过调用 ``model.initialize`` 来初始化 ``ddp`` 的通信组；（情况 B）

            .. code-block::

                trainer = Trainer(
                    ...
                    driver='deepspeed',
                    device=None
                )
                trainer.run()

            通过运行 ``deepspeed train.py`` 启动；

            3. 用户自己在外面初始化 ``deepspeed``，并且通过 ``deepspeed train.py`` 拉起，这时无论是多个进程的拉起和通信组的建立
            都由用户自己操作，我们只会在 ``driver.setup`` 的时候对 ``DeepSpeedDriver`` 设置一些必要的属性值；（情况 C）

            .. code-block::

                import deepspeed

                # 初始化
                model, _, _, _ = deepspeed.initialize(model, ...)

                trainer = Trainer(
                    ...
                    driver='deepspeed',
                    device=None
                )
                trainer.run()

            通过运行 ``deepspeed train.py`` 启动。

    :param model: 传入给 ``Trainer`` 的 ``model`` 参数。
    :param parallel_device: 用于分布式训练的 ``gpu`` 设备。
    :param is_pull_by_torch_run: 标志当前的脚本的启动是否由 ``python -m torch.distributed.launch`` 启动的。
    :param fp16: 是否开启 fp16 训练。
    :param deepspeed_kwargs: 
        * *strategy* -- 使用 ZeRO 优化的策略，默认为 ``deepspeed``；目前仅支持以下值：

            * ``deepspeed`` -- 使用 ZeRO 的第二阶段，等同于 ``deepspeed_stage_2``；
            * ``deepspeed_stage_1`` -- 使用 ZeRO 的第一阶段，仅将 ``optimizer`` 的状态分散到不同设备上；
            * ``deepspeed_stage_2`` -- 使用 ZeRO 的第二阶段，将 ``optimizer`` 和 **梯度** 分散到不同设备上；
            * ``deepspeed_stage_2_offload`` -- 使用 ZeRO 的第二阶段，并且借助 cpu 的内存来进一步节约显存；
            * ``deepspeed_stage_3`` -- 使用 ZeRO 的第三阶段，将 ``optimizer`` 、**梯度** 和 **模型** 分散到不同设备上；
            * ``deepspeed_stage_3_offload`` -- 使用 ZeRO 的第三阶段，并且借助 cpu 的内存来进一步节约显存；
            * ``deepspeed_stage_3_offload_nvme`` -- 使用 ZeRO 的第三阶段，并且借助 NVMe 硬盘来进一步节约显存；
        * *logging_level* -- ``deepspeed`` 库的日志等级，默认为 **logging.ERROR**。
        * *config* -- ``deepspeed`` 的各项设置；**FastNLP** 允许用户传入自己的设置以增强灵活性，但这会使参数
          中的 ``optimizer`` 、``strategy`` 、 ``fp16`` 等失效，即当这个参数存在时，**FastNLP** 会用该参数覆盖
          其它的设置。
    :kwargs:
        * *accumulation_steps* -- 即在 :class:`~fastNLP.core.controllers.Trainer` 传入的 ``accumulation_steps`` 。 deepspeed 会将 ``config`` 的
          ``gradient_accumulation_steps`` 设置为该值。
        * *train_dataloader* -- 即在 :class:`~fastNLP.core.controllers.Trainer` 传入的 ``train_dataloader`` 。 ``deepspeed`` 需要通过它来获取
          数据的 ``batch_size`` 用于设置 ``train_micro_batch_size_per_gpu`` 。如果没有传入的话，则会设置为 **1** 。
        * *wo_auto_param_call* (``bool``) -- 是否关闭在训练时调用我们的 ``auto_param_call`` 函数来自动匹配 batch 和前向函数的参数的行为

        .. note::

            关于该参数的详细说明，请参见 :class:`~fastNLP.core.controllers.Trainer` 中的描述；函数 ``auto_param_call`` 详见 :func:`fastNLP.core.utils.auto_param_call`。
    """
    # TODO fp16 load_config
    def __init__(
        self,
        model,
        parallel_device: Union[List["torch.device"], "torch.device"],
        is_pull_by_torch_run = False,
        fp16: bool = False,
        deepspeed_kwargs: Dict = None,
        **kwargs
    ):
        assert _NEED_IMPORT_DEEPSPEED, "Deepspeed is not imported."
        kwargs.pop("torch_kwargs", None)
        self._ds_kwargs = deepspeed_kwargs
        TorchDriver.__init__(self, model=model, fp16=False, torch_kwargs=deepspeed_kwargs, **kwargs)
        self.fp16 = fp16

        # 如果用户自己在外面初始化 DDP，那么其一定是通过 python -m torch.distributed.launch 拉起的；
        self.is_pull_by_torch_run = is_pull_by_torch_run
        self.parallel_device = parallel_device
        if not is_pull_by_torch_run and parallel_device is None:
            raise ValueError(
                "Parameter `parallel_device` can not be None when using `TorchDeepSpeedDriver`. This error is caused "
                "when your value of parameter `device` is `None` in your `Trainer` instance.")

        # 注意我们在 initialize_torch_driver 中的逻辑就是如果是 is_pull_by_torch_run，那么我们就直接把 parallel_device 置为当前进程的gpu；
        if is_pull_by_torch_run:
            self.model_device = parallel_device
        else:
            # 我们的 model_device 一定是 torch.device，而不是一个 list；
            self.model_device = parallel_device[self.local_rank]

        # 如果用户自己在外面初始化了 deepspeed；
        self.outside_ddp = False
        if dist.is_initialized() and FASTNLP_DISTRIBUTED_CHECK not in os.environ and \
                "fastnlp_torch_launch_not_ddp" not in os.environ:
            # 如果用户自己在外面初始化了 deepspeed，那么我们要求用户传入的模型一定是已经由 DeepSpeedEngine 包裹后的模型；
            if not isinstance(model, DeepSpeedEngine):
                raise RuntimeError(
                    "It is not allowed to input a normal model instead of `DeepSpeedEngine` when"
                    "you initialize the ddp process out of our control.")

            self.outside_ddp = True
            self.config = model.config
            self.model_device = None

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

        self.output_from_new_proc = kwargs.get("output_from_new_proc", "only_error")
        assert isinstance(self.output_from_new_proc, str), "Parameter `output_from_new_proc` can only be `str` type."
        if self.output_from_new_proc not in {"all", "ignore", "only_error"}:
            os.makedirs(name=self.output_from_new_proc, exist_ok=True)
            self.output_from_new_proc = os.path.abspath(self.output_from_new_proc)

        self._has_setup = False  # 设置这一参数是因为 evaluator 中也会进行 setup 操作，但是显然是不需要的也不应该的；
        self._has_ddpwrapped = False  # 判断传入的模型是否经过 _has_ddpwrapped 包裹；
        self.accumulation_steps = kwargs.get("accumulation_steps", 1)
        # 获取 batch_size 以设置 train_micro_batch_size_per_gpu 参数
        train_dl = kwargs.get("train_dataloader", None)
        if train_dl is not None:
            self.train_micro_batch_size = self.get_dataloader_args(train_dl).batch_size
        else:
            logger.warning("No `train_dataloader` found, and we will set `train_micro_batch_size_per_gpu`"
                        "to 1 for deepspeed configuration.")
            self.train_micro_batch_size = 1

        self.strategy = self._ds_kwargs.get("strategy", "deepspeed")
        deepspeed_logging_level = self._ds_kwargs.get("logging_level", logging.ERROR)
        deepspeed.utils.logging.logger.setLevel(deepspeed_logging_level)

    @staticmethod
    def _check_optimizer_legality(optimizers):
        for each_optimizer in optimizers:
            if not isinstance(each_optimizer, (Optimizer, DeepSpeedOptimizer)):
                raise TypeError(f"Each optimizer of parameter `optimizers` should be 'Optimizer' or "
                                f"'DeepSpeedOptimizer'type, not {type(each_optimizer)}.")

    def setup(self):
        r"""
        准备分布式环境，该函数主要做以下两件事情：

            1. 开启多进程，每个 gpu 设备对应单独的一个进程；
            2. 使用 ``deepspeed.initialize`` 包裹模型；
        """
        if len(self.optimizers) != 1:
            raise ValueError("Multi optimizers is not supported for `DeepSpeedDriver` right now.")
        if self._has_setup:
            return
        self._has_setup = True
        self.setup_config()
        # 如果用户需要使用多机模式，那么一定进入到这里；
        if self.is_pull_by_torch_run:
            if self.outside_ddp:
                self.world_size = dist.get_world_size()
                self.global_rank = dist.get_rank()
            else:
                # dist.get_world_size() 只能在 dist.init_process_group 初始化之后进行调用；
                self.world_size = int(os.environ.get("WORLD_SIZE"))
                self.global_rank = int(os.environ.get("RANK"))
                logger.info(f"World size: {self.world_size}, Global rank: {self.global_rank}")

                if not dist.is_initialized():
                    deepspeed.init_distributed("nccl", distributed_port=self.master_port)

                os.environ["fastnlp_torch_launch_not_ddp"] = "yes"

        # 进入到这里的情况时：
        # dist.is_initialized 一定为 False；
        # 一定是单机；
        # self.parallel_device 一定是 List[torch.device]；
        else:
            if not dist.is_initialized():
                # 这里主要的问题在于要区分 rank0 和其它 rank 的情况；
                self.world_size = len(self.parallel_device)
                self.open_subprocess()
                self.global_rank = self.local_rank  # rank 一定是通过环境变量去获取的；
                deepspeed.init_distributed("nccl", distributed_port=self.master_port)
            # 用户在这个 trainer 前面又初始化了一个 trainer，并且使用的是 DeepSpeedDriver；
            else:
                # 如果 `dist.is_initialized() == True`，那么说明 DeepSpeedDriver 在之前已经初始化并且已经 setup 过一次，那么我们需要保证现在
                #  使用的（即之后的）DeepSpeedDriver 的设置和第一个 DeepSpeedDriver 是完全一样的；
                pre_num_processes = int(os.environ[FASTNLP_DISTRIBUTED_CHECK])
                if pre_num_processes != len(self.parallel_device):
                    raise RuntimeError(
                        "Notice you are using `DeepSpeedDriver` after one instantiated `DeepSpeedDriver`, it is not"
                        "allowed that your second `DeepSpeedDriver` has a new setting of parameters "
                        "`num_nodes` and `num_processes`.")
                self.world_size = dist.get_world_size()
                self.global_rank = dist.get_rank()

        if not self.outside_ddp:
            torch.cuda.set_device(self.model_device)
            # 不加 dist.broadcast_object_list 会发生设备在 4,5 但是模型会同步到 0,1 的情况
            # 原因未知
            dist.broadcast_object_list(["test"], 0, None)
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
        
        # 设置 deepspeed
        if not isinstance(self.model, DeepSpeedEngine):
            model=_DeepSpeedWrappingModel(self.model, self.fp16)
            model_parameters = filter(lambda p: p.requires_grad, model.parameters())
            self.model, ds_optimizer, _, _ = deepspeed.initialize(
                args=argparse.Namespace(device_rank=self.model_device.index),
                model=model,
                optimizer=self.optimizers[0],
                model_parameters=model_parameters,
                config=self.config,
                dist_init_required=False
            )
            self._optimizers = [ds_optimizer]

            if self.config.get("activation_checkpointing"):
                checkpoint_config = self.config["activation_checkpointing"]
                deepspeed.checkpointing.configure(
                    mpu_=None,
                    partition_activations=checkpoint_config.get("partition_activations"),
                    contiguous_checkpointing=checkpoint_config.get("contiguous_memory_optimization"),
                    checkpoint_in_cpu=checkpoint_config.get("cpu_checkpointing"),
                    profile=checkpoint_config.get("profile"),
                )

            self._has_ddpwrapped = True

    def setup_config(self):

        self.config = self._ds_kwargs.get("config")
        if self.config is not None:
            logger.warning("Notice that you have defined a configuration for deepspeed and parameters like"
                        "`optimizers`, `strategy` and `fp16` may not take effects.")
            return

        if self.strategy == "deepspeed":
            self.config = _create_default_config(stage=2)
        elif self.strategy == "deepspeed_stage_1":
            self.config = _create_default_config(stage=1)
        elif self.strategy == "deepspeed_stage_2":
            self.config = _create_default_config(stage=2)
        elif self.strategy == "deepspeed_stage_2_offload":
            self.config = _create_default_config(stage=2, offload_optimizer=True)
        elif self.strategy == "deepspeed_stage_3":
            self.config = _create_default_config(stage=3)
        elif self.strategy == "deepspeed_stage_3_offload":
            self.config = _create_default_config(
                stage=3,
                offload_optimizer=True,
                offload_parameters=True,
            )
        elif self.strategy == "deepspeed_stage_3_offload_nvme":
            self.config = _create_default_config(
                stage=3,
                offload_optimizer=True,
                offload_parameters=True,
                remote_device="nvme",
                offload_params_device="nvme",
                offload_optimizer_device="nvme",
            )
        else:
            raise ValueError(f"Unknown deepspeed strategy {self.strategy}.")

        # 设置成 max_int 防止 deepspeed 的输出干扰 fastnlp 的输出
        self.config.setdefault("steps_per_print", 2147483647)
        self.config["gradient_accumulation_steps"] = self.accumulation_steps
        self.config.setdefault("train_micro_batch_size_per_gpu", self.train_micro_batch_size)

        if self.fp16:
            if "fp16" not in self.config:
                # FP16 is a DeepSpeed standalone AMP implementation
                logger.debug("Enabling DeepSpeed FP16.")
                # TODO 这部分是否可以像 pytorch-lightning 那样给用户定制
                self.config["fp16"] = {
                    "enabled": True,
                    "loss_scale": 0,
                    "initial_scale_power": True,
                    "loss_scale_window": 1000,
                    "hysteresis": 2,
                    "min_loss_scale": 1,
                }
            elif "amp" not in self.config:
                logger.debug("Enabling DeepSpeed APEX Implementation.")
                self.config["amp"] = {"enabled": True, "opt_level": "O1"}

    def zero_grad(self):
        """
        进行梯度置零操作；由于 :meth:`DeepSpeedEngine.step` 包含了 :meth:`zero_step` 的功能，因此该接口实际无意义。
        """
        # DeepSpeedEngine.step 包含了 zero_grad 功能
        pass

    def backward(self, loss):
        """
        对 ``loss`` 进行反向传播
        """
        self.model.backward(loss)

    def step(self):
        """
        更新模型的参数
        """
        self.model.step()

    def get_model_no_sync_context(self):
        r"""
        :return: 一个 ``context`` 上下文环境，用于关闭各个进程之间的同步；在 ``deepspeed`` 中，返回一个空的上下文
        """
        # 注意此时的 model 是 "DistributedDataParallel" 对象；
        return nullcontext

    def save_model(self, filepath: Union[str, Path], only_state_dict: bool = False, **kwargs):
        """
        保存的模型到 ``filepath`` 中。

        :param filepath: 文件路径
        :param only_state_dict: 是否只保存权重；在 ``DeepSpeedDriver`` 中该参数无效。
        :param kwargs: 需要传入 **deepspeed** 模型 :meth:`save_checkpoint` 的其它参数。
        :return:
        """
        # deepspeed engine 要求在每个 rank 都调用 save_checkpoint，故去掉了 rank_zero_call 装饰器
        if self.stage_3:
            logger.rank_zero_warning(
                "When saving the DeepSpeed Stage 3 checkpoint, "
                "each worker will save a shard of the checkpoint within a directory. "
                # TODO check一下
                # "If a single file is required after training, "
                # "see https://pytorch-lightning.readthedocs.io/en/latest/advanced/advanced_gpu.html#"
                # "deepspeed-zero-stage-3-single-file for instructions."
            )
        if not only_state_dict:
            logger.rank_zero_warning("Only saving state dict is not allowed for `DeepSpeedDriver`. We will save its "
                        "checkpoint for you instead.")
        self.model.save_checkpoint(filepath, **kwargs)

    def load_model(self, filepath: Union[Path, str], only_state_dict: bool = False, **kwargs):
        """
        从 ``filepath`` 中加载权重并赋值到当前 driver 的模型上。

        :param filepath: 加载权重或模型的路径
        :param load_state_dict: 保存的内容是否只是权重；在 ``DeepSpeedDriver`` 中该参数无效。
        :param kwargs: 需要传入 **deepspeed** 模型 :meth:`load_checkpoint` 的其它参数。
        :return:
        """
        if not only_state_dict:
            logger.warning("Only loading state dict is not allowed for `DeepSpeedDriver`. We will load its "
                        "checkpoint for you instead.")
        self.model.load_checkpoint(filepath, **kwargs)

    def save_checkpoint(self, folder: Path, states: Dict, dataloader, only_state_dict: bool = True, should_save_model: bool = True, **kwargs):
        r"""
        断点重训的保存函数，该函数会负责保存 **优化器** 、 **sampler** 和 **fp16** 的状态，以及 **模型** （若 ``should_save_model`` 为 ``True``）

        :param folder: 保存断点重训的状态的文件夹；:meth:`save_checkpoint` 函数应该在该路径下面下面新增名为 ``FASTNLP_CHECKPOINT_FILENAME`` 与
            ``FASTNLP_MODEL_FILENAME`` （如果 ``should_save_model`` 为 ``True`` ）的文件。把 model 相关的内容放入到 ``FASTNLP_MODEL_FILENAME`` 文件
            中，将传入的 ``states`` 以及自身产生其它状态一并保存在 ``FASTNLP_CHECKPOINT_FILENAME`` 里面。
        :param states: 由 :class:`~fastNLP.core.controllers.Trainer` 传入的一个字典，其中已经包含了为了实现断点重训所需要保存的其它对象的状态。
        :param dataloader: 正在使用的 dataloader。
        :param only_state_dict: 是否只保存模型的参数，当 ``should_save_model`` 为 ``False`` ，该参数无效。
        :param should_save_model: 是否应该保存模型，如果为 ``False`` ，Driver 将不负责 model 的保存。
        """
        # deepspeed engine 要求在每个 rank 都调用 save_checkpoint，故去掉了 rank_zero_call 装饰器
        # 1. 保存 sampler 的状态
        num_consumed_batches = states.pop('num_consumed_batches')
        states['sampler_states'] = self.get_sampler_state(dataloader, num_consumed_batches)

        # 2. 保存模型的状态；
        if not should_save_model:
            logger.rank_zero_warning("Saving checkpoint without model is not allowed for `DeepSpeedDriver`, "
                                    "so we will still save the model for you.")

        self.model.save_checkpoint(Path(folder).joinpath(FASTNLP_CHECKPOINT_FILENAME),
                                    client_state=states)

    def load_checkpoint(self, folder: Path, dataloader, only_state_dict: bool = True, should_load_model: bool = True, **kwargs) -> Dict:
        r"""
        断点重训的加载函数，该函数会负责读取数据，并且恢复 **优化器** 、**sampler** 、 **fp16** 的状态和 **模型** （如果 ``should_load_model`` 为 True）以及其它
        在 :meth:`save_checkpoint` 函数中执行的保存操作，然后将一个 state 字典返回给 :class:`~fastNLP.core.controllers.Trainer` （ 内容为 :meth:`save_checkpoint` 
        接受到的 ``states`` ）。

        该函数应该在所有 rank 上执行。

        :param folder: 读取该 folder 下的 ``FASTNLP_CHECKPOINT_FILENAME`` 文件与 ``FASTNLP_MODEL_FILENAME``
            （如果 should_load_model 为True）。
        :param dataloader: 当前给定 dataloader，需要根据保存的 dataloader 状态合理设置。若该值为 ``None`` ，则不需要返回 ``'dataloader'``
            以及 ``'batch_idx_in_epoch'`` 这两个值。
        :param only_state_dict: 是否仅读取模型的 state_dict ，当 ``should_save_model`` 为 ``False`` ，该参数无效。如果为 ``True`` ，说明保存的内容为权重；如果为
            False 说明保存的是模型，但也是通过当前 Driver 的模型去加载保存的模型的权重，而不是使用保存的模型替换当前模型。
        :param should_load_model: 是否应该加载模型，如果为 ``False`` ，Driver 将不负责加载模型。若该参数为 ``True`` ，但在保存的状态中没有
            找到对应的模型状态，则报错。
        :return: :meth:`save_checkpoint` 函数输入的 ``states`` 内容。除此之外，还返回的内容有：

            * *dataloader* -- 根据传入的 ``dataloader`` 与读取出的状态设置为合理状态的 dataloader。在当前 ``dataloader`` 样本数与读取出的 sampler 样本数
              不一致时报错。
            * *batch_idx_in_epoch* -- :class:`int` 类型的数据，表明当前 epoch 进行到了第几个 batch 。请注意，该值不能仅通过保存的数据中读取的，因为前后两次运行的
              ``batch_size`` 可能有变化，而应该符合以下等式::

                返回的 dataloader 还会产生的 batch 数量 + batch_idx_in_epoch = 原来不断点训练时的 batch 的总数
              
              由于 ``返回的 dataloader 还会产生的batch数`` 在 ``batch_size`` 与 ``drop_last`` 参数给定的情况下，无法改变，因此只能通过调整 ``batch_idx_in_epoch``
              这个值来使等式成立。一个简单的计算原则如下：

                * drop_last 为 ``True`` 时，等同于 floor(sample_in_this_rank/batch_size) - floor(num_left_samples/batch_size)；
                * drop_last 为 ``False`` 时，等同于 ceil(sample_in_this_rank/batch_size) - ceil(num_left_samples/batch_size)。
        """
        # 1. 加载模型状态；
        if not should_load_model:
            logger.rank_zero_warning("Loading checkpoint without model is not allowed for `DeepSpeedDriver`, "
                                    "so we will still load the model for you.")
        load_path, states = self.model.load_checkpoint(folder.joinpath(FASTNLP_CHECKPOINT_FILENAME))
        if load_path is None:
            raise RuntimeError(f"Failed to load checkpoint from path: {str(folder)}")

        # 2.恢复 sampler 的状态
        sampler_states = states.pop('sampler_states')
        states_ret = self.load_sampler_state(dataloader, sampler_states)
        states.update(states_ret)

        return states

    @property
    def stage_3(self) -> bool:
        """
        判断是否为第三阶段的 ZeRO 优化
        """
        return self.config.get("zero_optimization") and self.config.get("zero_optimization").get("stage") == 3
from typing import Optional, Union, Callable, Dict, Tuple, Sequence, List
from .torch_driver import TorchDriver
from .utils import _create_default_config
from fastNLP.core.utils import auto_param_call
from fastNLP.core.utils.utils import _get_fun_msg
from fastNLP.core.samplers import ReproducibleBatchSampler, ReproducibleSampler, re_instantiate_sampler, \
    ReproduceBatchSampler
from fastNLP.core.log import logger
from fastNLP.envs.imports import _NEED_IMPORT_TORCH, _NEED_IMPORT_DEEPSPEED

if _NEED_IMPORT_TORCH:
    import pytorch_lightning
    import torch
    from torch.nn import DataParallel
    
if _NEED_IMPORT_DEEPSPEED:
    import deepspeed

__all__ = [
    "DeepSpeedDriver",
]

class DeepSpeedDriver(TorchDriver):
    def __init__(self, model, fp16, strategy, **kwargs):
        super(DeepSpeedDriver, self).__init__(model, fp16)

        self.strategy = strategy

    def setup(self):

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
        for i, optimizer in enumerate(self.optimizers):
            # TODO 多个 optimizer
            engine, optimizer_ds, _, _ = deepspeed.initialize(
                model=self.model,
                optimizer=optimizer,
                config=self.config
            )
            self._optimizers[i] = optimizer_ds
        self.model = engine

        self._set_deepspeed_activation_checkpointing()

    def model_call(self, batch, fn: Callable, signature_fn: Optional[Callable]) -> Dict:
        if isinstance(batch, Dict) and not self.wo_auto_param_call:
            return auto_param_call(fn, batch, signature_fn=signature_fn)
        else:
            return fn(batch)

    def get_model_call_fn(self, fn: str) -> Tuple:
        if hasattr(self.model, fn):
            fn = getattr(self.model, fn)
            if not callable(fn):
                raise RuntimeError(f"The `{fn}` attribute is not `Callable`.")
            logger.debug(f'Use {_get_fun_msg(fn, with_fp=False)}...')
            return fn, None
        elif fn in {"train_step", "evaluate_step"}:
            logger.debug(f'Use {_get_fun_msg(self.model.forward, with_fp=False)}...')
            return self.model, self.model.forward
        else:
            raise RuntimeError(f"There is no `{fn}` method in your {type(self.model)}.")

    def set_dist_repro_dataloader(self, dataloader,
                                  dist: Union[str, ReproducibleBatchSampler, ReproducibleSampler] = None,
                                  reproducible: bool = False):
        return dataloader
        # 如果 dist 为 ReproducibleBatchSampler, ReproducibleIterator 说明是在断点重训时 driver.load_checkpoint 函数调用；
        if isinstance(dist, ReproducibleBatchSampler):
            return replace_batch_sampler(dataloader, dist)
        elif isinstance(dist, ReproducibleSampler):
            return replace_sampler(dataloader, dist)

        # 如果 dist 为 str 或者 None，说明是在 trainer 初试化时调用；
        args = self.get_dataloader_args(dataloader)
        if isinstance(args.batch_sampler, ReproducibleBatchSampler):
            batch_sampler = re_instantiate_sampler(args.batch_sampler)
            return replace_batch_sampler(dataloader, batch_sampler)
        elif isinstance(args.sampler, ReproducibleSampler):
            sampler = re_instantiate_sampler(args.sampler)
            return replace_sampler(dataloader, sampler)

        if reproducible:
            if type(args.batch_sampler) is TorchBatchSampler:
                if type(args.sampler) is TorchRandomSampler:
                    if getattr(args.sampler, '_num_samples', None) is None \
                            and getattr(args.sampler, 'replacements', False) is False \
                            and getattr(args.sampler, 'generator', None) is None:
                        # 如果本来就是随机的，并且没有定制，直接替换掉吧。
                        sampler = RandomSampler(args.sampler.data_source, shuffle=True)
                        logger.debug("Replace torch RandomSampler into fastNLP RandomSampler.")
                        return replace_sampler(dataloader, sampler)
                elif type(args.sampler) is TorchSequentialSampler:
                    # 需要替换为不要 shuffle 的。
                    sampler = RandomSampler(args.sampler.data_source, shuffle=False)
                    logger.debug("Replace torch SequentialSampler into fastNLP RandomSampler.")
                    return replace_sampler(dataloader, sampler)
            batch_sampler = ReproduceBatchSampler(
                batch_sampler=args.batch_sampler,
                batch_size=args.batch_size,
                drop_last=args.drop_last
            )
            return replace_batch_sampler(dataloader, batch_sampler)
        else:
            return dataloader

    def unwrap_model(self):
        r"""
        :return: 返回原本的模型，例如没有被 ``DataParallel`` 包裹；
        """
        if isinstance(self.model, deepspeed.DeepSpeedEngine):
            print(type(self.model.module), self.model.module)
            return self.model.module
        if isinstance(self.model, torch.nn.DataParallel) or \
                isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module
        else:
            return self.model

    @property
    def data_device(self):
        r"""
        注意单卡模式下使用 ``driver.data_device`` 等价于使用 ``driver.model_device``；
        """
        return self.model_device

    def is_distributed(self):
        r"""
        :return: 返回当前使用的 driver 是否是分布式的 driver，对于 ``TorchSingleDriver`` 来说直接返回 ``False``；
        """
        return False

    def _set_deepspeed_activation_checkpointing(self):
        if self.config.get("activation_checkpointing"):
            checkpoint_config = self.config["activation_checkpointing"]
            deepspeed.checkpointing.configure(
                mpu_=None,
                partition_activations=checkpoint_config.get("partition_activations"),
                contiguous_checkpointing=checkpoint_config.get("contiguous_memory_optimization"),
                checkpoint_in_cpu=checkpoint_config.get("cpu_checkpointing"),
                profile=checkpoint_config.get("profile"),
            )
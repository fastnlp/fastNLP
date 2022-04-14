import os
from typing import Dict, Union, Callable, Tuple, Optional
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch
    from torch.nn import DataParallel
    from torch.nn.parallel import DistributedDataParallel

__all__ = [
    'TorchSingleDriver'
]

from .torch_driver import TorchDriver
from fastNLP.core.drivers.torch_driver.utils import replace_sampler, replace_batch_sampler
from fastNLP.core.utils import auto_param_call
from fastNLP.core.samplers import ReproducibleBatchSampler, ReproducibleSampler, re_instantiate_sampler, RandomBatchSampler
from fastNLP.core.log import logger


class TorchSingleDriver(TorchDriver):
    r"""
    用于 cpu 和 单卡 gpu 运算；
    """
    def __init__(self, model, device: "torch.device", fp16: bool = False, **kwargs):
        if isinstance(model, DistributedDataParallel):
            raise ValueError("`DistributedDataParallel` is not supported in `TorchSingleDriver`")

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices == "":
            device = torch.device("cpu")
            logger.info("You have set `CUDA_VISIBLE_DEVICES` to '' in system environment variable, and we are gonna to"
                        "use `cpu` instead of `gpu` device.")

        super(TorchSingleDriver, self).__init__(model, fp16=fp16, **kwargs)

        if device is None:
            raise ValueError("Parameter `device` can not be None in `TorchSingleDriver`.")

        self.model_device = device

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

    def setup(self):
        if self.model_device is not None:
            self.model.to(self.model_device)

    def model_call(self, batch, fn: Callable, signature_fn: Optional[Callable]) -> Dict:
        if isinstance(batch, Dict) and not self.wo_auto_param_call:
            return auto_param_call(fn, batch, signature_fn=signature_fn)
        else:
            return fn(batch)

    def get_model_call_fn(self, fn: str) -> Tuple:
        if isinstance(self.model, DataParallel):
            model = self.unwrap_model()
            if hasattr(model, fn):
                logger.warning("Notice your model is a `DataParallel` model. And your model also implements the "
                               f"`{fn}` method, which we can not call actually, we will"
                               " call `forward` function instead of `train_step` and you should note that.")

            elif fn not in {"train_step", "evaluate_step"}:
                raise RuntimeError(f"There is no `{fn}` method in your model. And also notice that your model is a "
                                   f"`DataParallel` model, which means that we will only call model.forward function "
                                   f"when we are in forward propagation.")

            return self.model, model.forward
        else:
            if hasattr(self.model, fn):
                fn = getattr(self.model, fn)
                if not callable(fn):
                    raise RuntimeError(f"The `{fn}` attribute is not `Callable`.")
                return fn, None
            elif fn in {"train_step", "evaluate_step"}:
                return self.model, self.model.forward
            else:
                raise RuntimeError(f"There is no `{fn}` method in your model.")

    def set_dist_repro_dataloader(self, dataloader, dist: Union[str, ReproducibleBatchSampler, ReproducibleSampler]=None,
                                  reproducible: bool = False):

        # 如果 dist 为 ReproducibleBatchSampler, ReproducibleIterator 说明是在断点重训时 driver.load 函数调用；
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
            batch_sampler = RandomBatchSampler(
                batch_sampler=args.batch_sampler,
                batch_size=args.batch_size,
                drop_last=args.drop_last
            )
            return replace_batch_sampler(dataloader, batch_sampler)
        else:
            return dataloader

    def unwrap_model(self):
        if isinstance(self.model, torch.nn.DataParallel) or \
                isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return self.model.module
        else:
            return self.model

    @property
    def data_device(self):
        """
        单卡模式不支持 data_device；
        """
        return self.model_device

    def is_distributed(self):
        return False







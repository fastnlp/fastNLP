import os
from typing import Dict, Union
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
from fastNLP.core.samplers import ReproducibleBatchSampler, ReproducibleIterator
from fastNLP.core.log import logger
from fastNLP.core.samplers import re_instantiate_sampler


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

        if isinstance(model, DataParallel):
            model = self.unwrap_model()
            if hasattr(model, "train_step"):
                logger.warning("Notice your model is a `DataParallel` or `DistributedDataParallel` model. And your "
                               "model also implements the `train_step` method, which we can not call actually, we will"
                               " call `forward` function instead of `train_step` and you should note that.")
            self._train_step = self.model
            self._train_signature_fn = model.forward

            if hasattr(model, "validate_step"):
                logger.warning("Notice your model is a `DataParallel` or `DistributedDataParallel` model. And your "
                               "model also implements the `validate_step` method, which we can not call actually, "
                               "we will call `forward` function instead of `validate_step` and you should note that.")
            self._validate_step = self.model
            self._validate_signature_fn = model.forward

            if hasattr(model, "test_step"):
                logger.warning("Notice your model is a `DataParallel` or `DistributedDataParallel` model. And your "
                               "model also implements the `test_step` method, which we can not call actually, we will"
                               " call `forward` function instead of `test_step` and you should note that.")
            self._test_step = self.model
            self._test_signature_fn = model.forward
        else:
            if hasattr(self.model, "train_step"):
                self._train_step = self.model.train_step
                self._train_signature_fn = None
            else:
                self._train_step = self.model
                # 输入的模型是 `DataParallel` 或者 `DistributedDataParallel`，我们需要保证其 signature_fn 是正确的；
                model = self.unwrap_model()
                self._train_signature_fn = model.forward

            if hasattr(self.model, "validate_step"):
                self._validate_step = self.model.validate_step
                self._validate_signature_fn = None
            elif hasattr(self.model, "test_step"):
                self._validate_step = self.model.test_step
                self._validate_signature_fn = self.model.test_step
            else:
                self._validate_step = self.model
                model = self.unwrap_model()
                self._validate_signature_fn = model.forward

            if hasattr(self.model, "test_step"):
                self._test_step = self.model.test_step
                self._test_signature_fn = None
            elif hasattr(self.model, "validate_step"):
                self._test_step = self.model.validate_step
                self._test_signature_fn = self.model.validate_step
            else:
                self._test_step = self.model
                model = self.unwrap_model()
                self._test_signature_fn = model.forward

    def setup(self):
        if self.model_device is not None:
            self.model.to(self.model_device)

    def train_step(self, batch) -> Dict:
        # 如果 batch 是一个 Dict，我们就默认帮其做参数匹配，否则就直接传入到 `train_step` 函数中，让用户自己处理；
        if isinstance(batch, Dict):
            return auto_param_call(self._train_step, batch, signature_fn=self._train_signature_fn)
        else:
            return self._train_step(batch)

    def backward(self, loss):
        self.grad_scaler.scale(loss).backward()

    def step(self):
        for optimizer in self.optimizers:
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()

    def validate_step(self, batch) -> Dict:
        # 因为我们 Tester 的逻辑就是将所有的 metric 传给 tester，然后 tester 控制具体 metric 的 update 和 compute；因此不管用户是否
        # 实现 validate_step 函数，其都应该返回一个字典，具体使用哪些东西则是在 validate_batch_loop 中每一个具体的 metric 自己去拿的；
        if isinstance(batch, Dict):
            return auto_param_call(self._validate_step, batch, signature_fn=self._validate_signature_fn)
        else:
            return self._validate_step(batch)

    def test_step(self, batch) -> Dict:
        if isinstance(batch, Dict):
            return auto_param_call(self._test_step, batch, signature_fn=self._test_signature_fn)
        else:
            return self._test_step(batch)

    def set_dist_repro_dataloader(self, dataloader, dist: Union[str, ReproducibleBatchSampler, ReproducibleIterator],
                                  reproducible: bool = False, sampler_or_batch_sampler=None):
        if isinstance(dist, ReproducibleBatchSampler):
            return replace_batch_sampler(dataloader, dist)
        elif isinstance(dist, ReproducibleIterator):
            return replace_sampler(dataloader, dist)

        if reproducible:
            args = self.get_dataloader_args(dataloader)
            if isinstance(args.sampler, ReproducibleIterator):
                sampler = re_instantiate_sampler(args.sampler)
                return replace_sampler(dataloader, sampler)
            else:
                batch_sampler = ReproducibleBatchSampler(
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







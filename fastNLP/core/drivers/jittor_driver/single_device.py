from typing import Dict, Union

from .jittor_driver import JittorDriver
from fastNLP.core.utils import auto_param_call
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
from fastNLP.core.samplers import ReproducibleBatchSampler, ReproducibleIterator

if _NEED_IMPORT_JITTOR:
    import jittor

__all__ = [
    "JittorSingleDriver",
]

class JittorSingleDriver(JittorDriver):
    r"""
    用于 cpu 和 单卡 gpu 运算
    TODO: jittor 的 fp16
    """

    def __init__(self, model, device=None, fp16: bool = False, **kwargs):
        super(JittorSingleDriver, self).__init__(model, fp16)

        self.model_device = device

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        if hasattr(self.model, "train_step"):
            self._train_step = self.model.train_step
            self._train_signature_fn = None
        else:
            self._train_step = self.model
            model = self.unwrap_model()
            self._train_signature_fn = model.execute

        if hasattr(self.model, "validate_step"):
            self._validate_step = self.model.validate_step
            self._validate_signature_fn = None
        elif hasattr(self.model, "test_step"):
            self._validate_step = self.model.test_step
            self._validate_signature_fn = self.model.test_step
        else:
            self._validate_step = self.model
            model = self.unwrap_model()
            self._validate_signature_fn = model.execute

        if hasattr(self.model, "test_step"):
            self._test_step = self.model.test_step
            self._test_signature_fn = None
        elif hasattr(self.model, "validate_step"):
            self._test_step = self.model.validate_step
            self._test_signature_fn = self.model.validate_step
        else:
            self._test_step = self.model
            model = self.unwrap_model()
            self._test_signature_fn = model.execute

    def train_step(self, batch) -> Dict:
        if isinstance(batch, Dict):
            return auto_param_call(self._train_step, batch, signature_fn=self._train_signature_fn)
        else:
            return self._train_step(batch)

    def step(self):
        """
        jittor optimizers 的step函数可以传入参数loss
        此时会同时进行 zero_grad 和 backward
        为了统一，这里暂不使用这样的方式
        """
        for optimizer in self.optimizers:
            optimizer.step()

    def backward(self, loss):
        for optimizer in self.optimizers:
            optimizer.backward(loss)

    def zero_grad(self, set_to_none=False):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def validate_step(self, batch):
        if isinstance(batch, Dict):
            return auto_param_call(self._validate_step, batch, signature_fn=self._validate_signature_fn)
        else:
            return self._validate_step(batch)

    def test_step(self, batch):

        if isinstance(batch, Dict):
            return auto_param_call(self._test_step, batch, signature_fn=self._test_signature_fn)
        else:
            return self._test_step(batch)

    def unwrap_model(self):
        return self.model

    def is_distributed(self):
        return False

    def replace_sampler(self, dataloader, dist_sampler: Union[str, ReproducibleBatchSampler, ReproducibleIterator], reproducible: bool = False):
        # reproducible 的相关功能暂时没有实现
        if isinstance(dist_sampler, ReproducibleBatchSampler):
            raise NotImplementedError
            dataloader.batch_sampler = dist_sample
        if isinstance(dist_sampler, ReproducibleIterator):
            raise NotImplementedError  
            dataloader.batch_sampler.sampler = dist_sampler        

        if reproducible:
            raise NotImplementedError
            if isinstance(dataloader.batch_sampler.sampler, ReproducibleIterator):
                return dataloader
            elif isinstance(dataloader.batch_sampler, ReproducibleBatchSampler):
                return dataloader
            else:
                # TODO
                batch_sampler = ReproducibleBatchSampler(
                    batch_sampler=dataloader.batch_sampler,
                    batch_size=dataloader.batch_sampler.batch_size,
                    drop_last=dataloader.drop_last
                )
                dataloader.batch_sampler = batch_sampler
                return dataloader
        else:
            return dataloader

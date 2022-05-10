from typing import Dict, Union, Tuple, Callable, Optional

from .jittor_driver import JittorDriver
from fastNLP.core.utils import auto_param_call
from fastNLP.core.utils.utils import _get_fun_msg
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
from fastNLP.core.samplers import ReproducibleBatchSampler, ReproducibleSampler
from fastNLP.core.log import logger

if _NEED_IMPORT_JITTOR:
    import jittor

__all__ = [
    "JittorSingleDriver",
]

class JittorSingleDriver(JittorDriver):
    r"""
    ``Jittor`` 框架下用于 ``cpu`` 和单卡 ``gpu`` 运算的 ``Driver``。

    .. note::

        这是一个正在开发中的功能，敬请期待。

    .. todo::

        支持 cpu 和 gpu 的切换；
        实现断点重训中替换 dataloader 的 set_dist_repro_dataloader 函数

    """

    def __init__(self, model, device=None, fp16: bool = False, **kwargs):
        super(JittorSingleDriver, self).__init__(model, fp16)

        self.model_device = device

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def backward(self, loss):
        for optimizer in self.optimizers:
            optimizer.backward(loss)

    def zero_grad(self, set_to_none=False):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

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
            logger.debug(f'Use {_get_fun_msg(self.model.execute, with_fp=False)}...')
            return self.model, self.model.execute
        else:
            raise RuntimeError(f"There is no `{fn}` method in your {type(self.model)}.")

    def unwrap_model(self):
        return self.model

    def is_distributed(self):
        return False

    def set_dist_repro_dataloader(self, dataloader, dist: Union[str, ReproducibleBatchSampler, ReproducibleSampler],
                                  reproducible: bool = False, sampler_or_batch_sampler=None):
        # reproducible 的相关功能暂时没有实现
        if isinstance(dist, ReproducibleBatchSampler):
            raise NotImplementedError
            dataloader.batch_sampler = dist_sample
        if isinstance(dist, ReproducibleSampler):
            raise NotImplementedError  
            dataloader.batch_sampler.sampler = dist

        if reproducible:
            raise NotImplementedError
            if isinstance(dataloader.batch_sampler.sampler, ReproducibleSampler):
                return dataloader
            elif isinstance(dataloader.batch_sampler, RandomBatchSampler):
                return dataloader
            else:
                # TODO
                batch_sampler = RandomBatchSampler(
                    batch_sampler=dataloader.batch_sampler,
                    batch_size=dataloader.batch_sampler.batch_size,
                    drop_last=dataloader.drop_last
                )
                dataloader.batch_sampler = batch_sampler
                return dataloader
        else:
            return dataloader

    def setup(self):
        """
        使用单个 GPU 时，jittor 底层自动实现调配，无需额外操作
        """
        pass

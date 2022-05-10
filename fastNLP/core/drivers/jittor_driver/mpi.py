import os
from typing import Optional, Union, Callable, Dict, Tuple

from .jittor_driver import JittorDriver
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
from fastNLP.core.samplers import ReproducibleSampler

if _NEED_IMPORT_JITTOR:
    import jittor

__all__ = [
    "JittorMPIDriver",
]

class JittorMPIDriver(JittorDriver):
    """
    执行 ``Jittor`` 框架下分布式训练的 ``Driver``。

    .. note::

        这是一个正在开发中的功能，敬请期待。

    """
    def __init__(
        self,
        model,
        parallel_device: None,
        is_pull_by_jittor_run: bool = False,
        fp16: bool = False,
        **kwargs
    ):

        super(JittorMPIDriver, self).__init__(model, fp16=fp16, **kwargs)

        self.is_pull_by_jittor_run = is_pull_by_jittor_run
        self.parallel_device = parallel_device

        self.outside_mpi = False

    def setup(self):
        pass

    def configure_mpi(self):
        pass

    @property
    def world_size(self) -> int:
        return self._world_size

    @world_size.setter
    def world_size(self, size: int):
        self._world_size = size

    @property
    def global_rank(self) -> int:
        return self._global_rank

    @global_rank.setter
    def global_rank(self, rank: int) -> None:
        self._global_rank = rank

    @property
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    @property
    def data_device(self):
        if self.outside_mpi:
            return self._data_device
        return self.model_device

    def model_call(self, batch, fn: Callable, signature_fn: Optional[Callable]) -> Dict:
        pass

    def get_model_call_fn(self, fn: str) -> Tuple:
        pass

    def set_dist_repro_dataloader(self, dataloader, dist: Optional[Union[str, ReproducibleSampler]],
                                  reproducible: bool = False, sampler_or_batch_sampler=None):
        pass

    def backward(self, loss):
        self.grad_scaler.scale(loss).backward()

    def step(self):
        for optimizer in self.optimizers:
            self.grad_scaler.step(optimizer)
            self.grad_scaler.update()

    def is_global_zero(self):
        return self.global_rank == 0

    def get_model_no_sync_context(self):
        return self.model.no_sync

    def unwrap_model(self):
        pass

    def get_local_rank(self) -> int:
        return self.local_rank

    def barrier(self):
        pass

    def is_distributed(self):
        return True
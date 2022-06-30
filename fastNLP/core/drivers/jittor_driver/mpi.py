import os
from typing import Optional, Union, Callable, Dict, Tuple

from .jittor_driver import JittorDriver
from fastNLP.core.utils import auto_param_call
from fastNLP.core.utils.utils import _get_fun_msg
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
from fastNLP.core.samplers import ReproducibleBatchSampler, ReproducibleSampler
from fastNLP.core.log import logger

if _NEED_IMPORT_JITTOR:
    import jittor as jt

__all__ = [
    "JittorMPIDriver",
]

class JittorMPIDriver(JittorDriver):
    """
    执行 ``Jittor`` 框架下分布式训练的 ``Driver``。

    .. note::

        这是一个正在开发中的功能，敬请期待。

    .. todo:

        实现断点重训中替换 dataloader 的 set_dist_repro_dataloader 函数

    """
    def __init__(
        self,
        model,
        parallel_device: None,
        is_pull_by_jittor_run: bool = False,
        fp16: bool = False,
        jittor_kwargs: Dict = {},
        **kwargs
    ):

        super(JittorMPIDriver, self).__init__(model, fp16=fp16, jittor_kwargs=jittor_kwargs, **kwargs)
        raise NotImplementedError("MPI for Jittor is not supported right now.")

        self.is_pull_by_jittor_run = is_pull_by_jittor_run
        self.parallel_device = parallel_device

        self.outside_mpi = False

    def setup(self):
        self.__fork_with_mpi__()

    def __fork_with_mpi__(self):
        import sys
        if jt.in_mpi:
            # you can mult other process output
            if jt.rank != 0:
                sys.stdout = open("/dev/null", "w")
            return
        else:
            if self.parallel_device == -1:              # device 为 -1，那么默认使用全部的显卡
                raise NotImplementedError(f"Device={self.parallel_device}")
            elif type(self.parallel_device) is int:     # device 为 *int*: 将使用 ``device_id`` 为该值的 ``gpu`` 进行训练
                num_procs = 1
                devices = self.parallel_device
            elif type(self.parallel_device) is list:    # device 为 *list(int)*: 多于 1 个device，应当通过该种方式进行设定
                num_procs = len(self.parallel_device)
                devices = str(self.parallel_device)[1:-1]
            else:
                raise NotImplementedError(f"Device={self.parallel_device}")
            print(sys.argv)
            cmd = " ".join(["CUDA_VISIBLE_DEVICES='%s'" % devices, "mpirun", "-np", str(num_procs), sys.executable] + sys.argv)
            print("[RUN CMD]:", cmd)
            os.system(cmd)
            exit(0)

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
        return self.parallel_device

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

    def set_dist_repro_dataloader(self, dataloader, dist: Optional[Union[str, ReproducibleSampler]],
                                  reproducible: bool = False, sampler_or_batch_sampler=None):
        return dataloader

    def is_global_zero(self):
        return self.global_rank == 0

    def get_model_no_sync_context(self):
        return self.model.no_sync

    def unwrap_model(self):
        """
        返回训练使用的模型。
        """
        return self.model

    def get_local_rank(self) -> int:
        return self.local_rank

    def barrier(self):
        pass

    def is_distributed(self):
        """
        判断是否为分布式的 **Driver** ，在 ``JittorSingleDriver`` 中，返回 ``True``。
        """
        return True

    @property
    def data_device(self) -> str:
        """
        :return: 数据所在的设备；
        """
        return self.model_device
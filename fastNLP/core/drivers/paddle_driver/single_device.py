import os
import contextlib
from typing import Optional, Dict, Union, Callable, Tuple

from .paddle_driver import PaddleDriver
from .utils import replace_batch_sampler, replace_sampler
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.envs.env import USER_CUDA_VISIBLE_DEVICES
from fastNLP.core.utils import (
    auto_param_call,
    get_device_from_visible,
    get_paddle_gpu_str,
    get_paddle_device_id,
)
from fastNLP.core.utils.utils import _get_fun_msg
from fastNLP.core.samplers import (
    ReproducibleBatchSampler,
    ReproduceBatchSampler,
    ReproducibleSampler,
    RandomSampler,
    re_instantiate_sampler,
)
from fastNLP.core.log import logger

if _NEED_IMPORT_PADDLE:
    import paddle
    from paddle import DataParallel
    from paddle.fluid.reader import _DatasetKind

__all__ = [
    "PaddleSingleDriver",
]

class PaddleSingleDriver(PaddleDriver):
    """
    支持 paddle cpu 或单卡 gpu 训练的 driver
    """
    def __init__(self, model, device: Union[str, int], fp16: Optional[bool] = False, **kwargs):
        if isinstance(model, DataParallel):
            raise ValueError("`paddle.DataParallel` is not supported in `PaddleSingleDriver`")

        cuda_visible_devices = os.getenv(USER_CUDA_VISIBLE_DEVICES)
        if cuda_visible_devices == "":
            device = "cpu"
            logger.info("You have set `CUDA_VISIBLE_DEVICES` to '' in system environment variable, and we are gonna to"
                        "use `cpu` instead of `gpu` device.")

        super(PaddleSingleDriver, self).__init__(model, fp16=fp16, **kwargs)

        if device is None:
            raise ValueError("Parameter `device` can not be None in `PaddleSingleDriver`.")

        if device != "cpu":
            device_id = get_paddle_device_id(device)
            if cuda_visible_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices.split(",")[device_id]
        self.model_device = get_paddle_gpu_str(device)

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

    def setup(self):
        r"""
        该函数用来初始化训练环境，用于设置当前训练的设备，并将模型迁移到对应设备上。
        """
        if USER_CUDA_VISIBLE_DEVICES in os.environ:
            device = get_device_from_visible(self.data_device)
        else:
            device = self.data_device

        paddle.device.set_device(device)
        with contextlib.redirect_stdout(None):
            self.model.to(device)

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

    def set_dist_repro_dataloader(self, dataloader, dist: Union[str, ReproducibleBatchSampler, ReproducibleSampler]=None,
                                  reproducible: bool = False):

        # 暂时不支持iterableDataset
        assert dataloader.dataset_kind != _DatasetKind.ITER, \
                    "FastNLP does not support `IteratorDataset` now."
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
            if isinstance(args.sampler, paddle.io.RandomSampler):
                # 如果本来就是随机的，直接替换
                sampler = RandomSampler(args.sampler.data_source)
                logger.debug("Replace paddle RandomSampler into fastNLP RandomSampler.")
                return replace_sampler(dataloader, sampler)
            else:
                batch_sampler = ReproduceBatchSampler(
                    batch_sampler=args.batch_sampler,
                    batch_size=args.batch_size,
                    drop_last=args.drop_last
                )
                return replace_batch_sampler(dataloader, batch_sampler)
        else:
            return dataloader

    def unwrap_model(self):
        if isinstance(self.model, paddle.DataParallel):
            return self.model._layers
        else:
            return self.model

    @property
    def data_device(self):
        """
        返回数据所在的设备。由于单卡模式不支持 data_device，因此返回的是 model_device
        """
        return self.model_device

    def is_distributed(self):
        return False

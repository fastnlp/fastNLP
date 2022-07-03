from typing import Dict, Union, Tuple, Callable, Optional

from .jittor_driver import JittorDriver
from .utils import replace_batch_sampler, replace_sampler
from fastNLP.core.utils import auto_param_call
from fastNLP.core.utils.utils import _get_fun_msg
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
from fastNLP.core.samplers import ReproducibleBatchSampler, ReproducibleSampler, re_instantiate_sampler, \
    ReproduceBatchSampler
from fastNLP.core.samplers import RandomSampler
from fastNLP.core.log import logger

if _NEED_IMPORT_JITTOR:
    import jittor as jt
    from jittor.dataset import (
        RandomSampler as JittorRandomSampler,
        SequentialSampler as JittorSequentialSampler,
    )

__all__ = [
    "JittorSingleDriver",
]

class JittorSingleDriver(JittorDriver):
    r"""
    ``Jittor`` 框架下用于 ``cpu`` 和单卡 ``gpu`` 运算的 ``Driver``。

    :param model: 传入给 ``Trainer`` 的 ``model`` 参数；
    :param device: 训练和模型所在的设备，在 **Jittor** 中，应当为以下值之一：``[None, 'cpu', 'gpu', 'cuda']``；
        
        * 为 ``None`` 或 ``cpu`` 时
         表示在 ``cpu`` 上进行训练；
        * 为 ``gpu`` 或 ``cuda`` 时
         表示在显卡设备上进行训练；

    :param fp16: 是否开启 fp16；
    :param jittor_kwargs:
    """

    def __init__(self, model, device=None, fp16: bool = False, jittor_kwargs: Dict = None, **kwargs):
        if device not in [None, "cpu", "gpu", "cuda"]:
            raise RuntimeError("Parameter `device` should be one of [None, 'cpu', 'gpu', 'cuda'] .")
        super(JittorSingleDriver, self).__init__(model, fp16, jittor_kwargs=jittor_kwargs)

        self.model_device = device if device is not None else "cpu"

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

    def setup(self):
        r"""
        初始化训练环境；根据传入的 ``device`` 值设置模型的训练场景为 ``cpu`` 或 ``gpu``；
        """
        if self.model_device in ["cpu", None]:
            jt.flags.use_cuda = 0   # 使用 cpu
        else:
            jt.flags.use_cuda = 1   # 使用 cuda

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
        """
        返回训练使用的模型。
        """
        return self.model

    def is_distributed(self):
        """
        判断是否为分布式的 **Driver** ，在 ``JittorSingleDriver`` 中，返回 ``False``。
        """
        return False

    def set_dist_repro_dataloader(self, dataloader,
                                  dist: Union[str, ReproducibleBatchSampler, ReproducibleSampler] = None,
                                  reproducible: bool = False):
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
            if args.sampler is None:
                sampler = RandomSampler(args.dataset, args.shuffle)
                return replace_sampler(dataloader, sampler)
            elif type(args.sampler) is JittorRandomSampler:
                if getattr(args.sampler, '_num_samples', None) is None \
                        and getattr(args.sampler, 'rep', False) is False:
                    # 如果本来就是随机的，并且没有定制，直接替换掉吧。
                    sampler = RandomSampler(args.sampler.dataset, shuffle=True)
                    logger.debug("Replace jittor RandomSampler into fastNLP RandomSampler.")
                    return replace_sampler(dataloader, sampler)
            elif type(args.sampler) is JittorSequentialSampler:
                # 需要替换为不要 shuffle 的。
                sampler = RandomSampler(args.sampler.dataset, shuffle=False)
                logger.debug("Replace jittor SequentialSampler into fastNLP RandomSampler.")
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
        """
        返回训练使用的模型。
        """
        return self.model

    @property
    def data_device(self) -> str:
        """
        :return: 数据和模型所在的设备；
        """
        return self.model_device

import os
from typing import Dict, Union
from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW

if _NEED_IMPORT_ONEFLOW:
    import oneflow
    from oneflow.utils.data import SequentialSampler as OneflowSequentialSampler
    from oneflow.utils.data import BatchSampler as OneflowBatchSampler

__all__ = [
    "OneflowSingleDriver"
]

from .oneflow_driver import OneflowDriver
from fastNLP.core.drivers.oneflow_driver.utils import replace_sampler, replace_batch_sampler
from fastNLP.core.samplers import ReproducibleBatchSampler, ReproducibleSampler, re_instantiate_sampler, \
    ReproduceBatchSampler
from fastNLP.core.samplers import RandomSampler
from fastNLP.core.log import logger


class OneflowSingleDriver(OneflowDriver):
    r"""
    用于执行 ``oneflow`` 动态图 cpu 和 单卡 gpu 运算的 ``driver``；

    :param model: 传入给 ``Trainer`` 的 ``model`` 参数；
    :param device: oneflow.device，当前进程所使用的设备；
    :param fp16: 是否开启 fp16；目前动态图的单卡下该参数无效；
    :param oneflow_kwargs:
    """

    def __init__(self, model, device: "oneflow.device", fp16: bool = False, oneflow_kwargs: Dict = None, **kwargs):
        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if cuda_visible_devices == "":
            device = oneflow.device("cpu")
            logger.info("You have set `CUDA_VISIBLE_DEVICES` to '' in system environment variable, and we are gonna to"
                        "use `cpu` instead of `gpu` device.")

        super(OneflowSingleDriver, self).__init__(model, fp16=fp16, oneflow_kwargs=oneflow_kwargs, **kwargs)

        if device is None:
            logger.debug("device is not set, fastNLP will try to automatically get it.")
            try:
                device = next(model.parameters()).device
                assert isinstance(device, oneflow.device)
            except:
                raise ValueError("fastNLP cannot get device automatically, please set device explicitly.")

        self.model_device = device

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

    def setup(self):
        r"""
        将模型迁移到相应的设备上；
        """
        if self.model_device is not None:
            self.model.to(self.model_device)

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
            if type(args.batch_sampler) is OneflowBatchSampler:
                if type(args.sampler) is OneflowSequentialSampler:
                    # 需要替换为不要 shuffle 的。
                    sampler = RandomSampler(args.sampler.data_source, shuffle=False)
                    logger.debug("Replace oneflow SequentialSampler into fastNLP RandomSampler.")
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
        :return: 返回模型
        """
        return self.model

    @property
    def data_device(self):
        r"""
        :return: 数据和模型所在的设备；
        """
        return self.model_device

    def is_distributed(self):
        r"""
        :return: 返回当前使用的 driver 是否是分布式的 driver，在 ``OneflowSingleDriver`` 中返回 ``False``；
        """
        return False

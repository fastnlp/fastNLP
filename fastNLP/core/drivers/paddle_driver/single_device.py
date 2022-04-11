import os
from typing import Optional, Dict, Union

from .paddle_driver import PaddleDriver
from .utils import replace_batch_sampler, replace_sampler
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.envs.env import USER_CUDA_VISIBLE_DEVICES
from fastNLP.core.utils import (
    auto_param_call,
    get_paddle_gpu_str,
    get_paddle_device_id,
    paddle_move_data_to_device,
)
from fastNLP.core.samplers import ReproducibleBatchSampler, ReproducibleIterator, re_instantiate_sampler
from fastNLP.core.log import logger

if _NEED_IMPORT_PADDLE:
    import paddle
    from paddle.fluid.reader import _DatasetKind

__all__ = [
    "PaddleSingleDriver",
]

class PaddleSingleDriver(PaddleDriver):
    def __init__(self, model, device: str, fp16: Optional[bool] = False, **kwargs):
        super(PaddleSingleDriver, self).__init__(model, fp16=fp16, **kwargs)

        if device is None:
            raise ValueError("Parameter `device` can not be None in `PaddleSingleDriver`.")

        if isinstance(device, int):
            self.model_device = get_paddle_gpu_str(device)
        else:
            self.model_device = device

        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1

        if isinstance(model, paddle.DataParallel):
            # 注意这里的 unwrap_model 调用的是具体子类的方法；
            model = self.unwrap_model()
            if hasattr(model, "train_step"):
                logger.warning("Notice your model is a `paddle.DataParallel` model. And your model also "
                                "implements the `train_step` method, which we can not call actually, we will "
                                " call `forward` function instead of `train_step` and you should note that.")
            self._train_step = self.model
            self._train_signature_fn = model.forward

            if hasattr(model, "validate_step"):
                logger.warning("Notice your model is a `paddle.DataParallel` model. And your model also "
                                "implements the `validate_step` method, which we can not call actually, we "
                                "will call `forward` function instead of `validate_step` and you should note that.")
            self._validate_step = self.model
            self._validate_signature_fn = model.forward

            if hasattr(model, "test_step"):
                logger.warning("Notice your model is a `paddle.DataParallel` model. And your model also "
                               "implements the `test_step` method, which we can not call actually, we will "
                               "call `forward` function instead of `test_step` and you should note that.")
            self._test_step = self.model
            self._test_signature_fn = model.forward
        else:
            if hasattr(self.model, "train_step"):
                self._train_step = self.model.train_step
                self._train_signature_fn = None
            else:
                self._train_step = self.model
                # 输入的模型是 `DataParallel`，我们需要保证其 signature_fn 是正确的；
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
        device_id = get_paddle_device_id(self.model_device)
        device_id = os.environ[USER_CUDA_VISIBLE_DEVICES].split(",")[device_id]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
        paddle.device.set_device("gpu:0")
        self.model.to("gpu:0")

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
        if isinstance(batch, Dict):
            return auto_param_call(self._validate_step, batch, signature_fn=self._validate_signature_fn)
        else:
            return self._validate_step(batch)

    def test_step(self, batch) -> Dict:
        if isinstance(batch, Dict):
            return auto_param_call(self._test_step, batch, signature_fn=self._test_signature_fn)
        else:
            return self._test_step(batch)

    def move_data_to_device(self, batch: 'paddle.Tensor'):
        r"""
        将数据迁移到指定的机器上；batch 可能是 list 也可能 dict ，或其嵌套结构。
        在 Paddle 中使用可能会引起因与设置的设备不一致而产生的问题，请注意。
        在单卡时，由于 CUDA_VISIBLE_DEVICES 始终被限制在一个设备上，因此实际上只会迁移到 `gpu:0`

        :return: 将移动到指定机器上的 batch 对象返回；
        """
        return paddle_move_data_to_device(batch, "gpu:0")

    def set_dist_repro_dataloader(self, dataloader, dist: Union[str, ReproducibleBatchSampler, ReproducibleIterator],
                                  reproducible: bool = False, sampler_or_batch_sampler=None):
        # 暂时不支持IteratorDataset
        assert dataloader.dataset_kind != _DatasetKind.ITER, \
                "FastNLP does not support `IteratorDataset` now."
        if isinstance(dist, ReproducibleBatchSampler):
            return replace_batch_sampler(dataloader, dist)
        elif isinstance(dist, ReproducibleIterator):
            return replace_sampler(dataloader, dist)      

        if reproducible:
            args = self.get_dataloader_args(dataloader)
            if isinstance(args.sampler, ReproducibleIterator):
                sampler = re_instantiate_sampler(args.sampler)
                return replace_sampler(dataloader, sampler)
            elif isinstance(dataloader.batch_sampler, ReproducibleBatchSampler):
                batch_sampler = re_instantiate_sampler(dataloader.batch_sampler)
                return replace_batch_sampler(dataloader, batch_sampler)
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
        if isinstance(self.model, paddle.DataParallel):
            return self.model._layers
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

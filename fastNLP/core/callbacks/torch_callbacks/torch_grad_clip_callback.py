__all__ = [
    'TorchGradClipCallback'
]
from typing import Union, List
from ..callback import Callback
from ...drivers.torch_driver.fairscale import FairScaleDriver
from ...drivers.torch_driver import TorchDriver
from fastNLP.envs.imports import _NEED_IMPORT_FAIRSCALE
if _NEED_IMPORT_FAIRSCALE:
    from fairscale.nn import FullyShardedDataParallel

class TorchGradClipCallback(Callback):
    r"""
    在每次 :func:`optimizer.step` 之前对参数的梯度进行截断。

    :param clip_value: 将梯度限制到 [-clip_value, clip_value] 之间。``clip_value`` 应该为正数；
    :param clip_type: 应为 ``'norm'``, ``'value'`` 中的一个:

        1. 为 ``'norm'`` 时, 将梯度的范数限制在 [-clip_value, clip_value] 之间；
        2. 为 ``'value'`` 时，, 将梯度限制在 [-clip_value, clip_value] 之间，小于 ``-clip_value``
           的梯度被赋值为 ``-clip_value``，大于 ``clip_value`` 的梯度被赋值为 ``clip_value``；

    :param parameters: 参数，一般通过 :func:`model.parameters` 获得。
        如果为 ``None`` 则默认对 Trainer 的 optimizers 中所有参数进行梯度裁剪。
    """
    def __init__(self, clip_value:int=1, clip_type:str='norm',
                 parameters:Union["torch.Tensor", List["torch.Tensor"]]=None):
        super().__init__()

        from torch import nn
        if clip_type == 'norm':
            self.clip_fun = nn.utils.clip_grad_norm_
        elif clip_type == 'value':
            self.clip_fun = nn.utils.clip_grad_value_
        else:
            raise ValueError("Only supports `norm` or `value` right now.")
        if parameters is not None:
            self.parameters = list(parameters)
        else:
            self.parameters = None
        self.clip_value = clip_value
        self.clip_type = clip_type

    def on_after_trainer_initialized(self, trainer, driver):
        assert isinstance(driver, TorchDriver), f"Callback:{self.__class__.__name__} only supports torch " \
                                                             f"related drivers for now."
        parameters = []
        for optimizer in trainer.driver.optimizers:
            for param_group in optimizer.param_groups:
                parameters.extend(param_group['params'])
        self.parameters = parameters
        if isinstance(trainer.driver, FairScaleDriver):
            if isinstance(trainer.driver.model, FullyShardedDataParallel) and self.clip_type == 'norm':
                self.clip_fun = trainer.driver.model.clip_grad_norm_

        assert len(self.parameters), "There is no parameters need to be clipped."

    def on_before_optimizers_step(self, trainer, optimizers):
        for optimizer in trainer.driver.optimizers:
            trainer.driver.grad_scaler.unscale_(optimizer)
        self.clip_fun(self.parameters, self.clip_value)

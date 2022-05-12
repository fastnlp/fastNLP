__all__ = [
    'TorchGradClipCallback'
]
from typing import Union, List
from ..callback import Callback


class TorchGradClipCallback(Callback):
    r"""
    在每次 optimizer update 之前将 parameter 进行 clip 。

    :param clip_value: 将gradient 限制到[-clip_value, clip_value]。clip_value应该为正数
    :param clip_type: 支持'norm', 'value'两种:

        1. 'norm', 将gradient的norm rescale到[-clip_value, clip_value]
        2. 'value', 将gradient限制在[-clip_value, clip_value],
         小于-clip_value的gradient被赋值为-clip_value;大于clip_value的gradient被赋值为clip_value.

    :param None,torch.Tensor,List[torch.Tensor] parameters: 一般通过model.parameters()获得。
        如果为None则默认对 Trainer 的 optimizers 中所有参数进行梯度裁剪。
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

    def on_after_trainer_initialized(self, trainer, driver):
        assert 'torch' in driver.__class__.__name__.lower(), f"Callback:{self.__class__.__name__} only supports torch " \
                                                             f"related drivers for now."
        parameters = []
        for optimizer in trainer.driver.optimizers:
            for param_group in optimizer.param_groups:
                parameters.extend(param_group['params'])
        self.parameters = parameters
        assert len(self.parameters), "There is no parameters need to be clipped."

    def on_before_optimizers_step(self, trainer, optimizers):
        for optimizer in trainer.driver.optimizers:
            trainer.driver.grad_scaler.unscale_(optimizer)
        self.clip_fun(self.parameters, self.clip_value)

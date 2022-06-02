__all__ = [
    'TorchWarmupCallback'
]
import math
from typing import Union

from ..callback import Callback


class TorchWarmupCallback(Callback):
    r"""
    调整 learning rate 的 callback 。

    :param warmup: 如果warmup为int，则在该step之前，learning rate根据schedule的策略变化; 如果warmup为float，
        如0.1, 则前10%的step是按照schedule策略调整learning rate。
    :param schedule: 以哪种方式调整。

        1. linear: 前warmup的step上升到指定的learning rate(从Trainer中的optimizer处获取的), 后warmup的step下降到0；
        2. constant前warmup的step上升到指定learning rate，后面的step保持learning rate.
    """
    def __init__(self, warmup:Union[int, float]=0.1, schedule:str='constant'):
        super().__init__()
        self.warmup = max(warmup, 0.)

        self.initial_lrs = []  # 存放param_group的learning rate
        if schedule == 'constant':
            self.get_lr = self._get_constant_lr
        elif schedule == 'linear':
            self.get_lr = self._get_linear_lr
        else:
            raise RuntimeError("Only support 'linear', 'constant'.")

    def _get_constant_lr(self, progress):
        if progress <self.warmup:
            return progress /self.warmup
        return 1

    def _get_linear_lr(self, progress):
        if progress <self.warmup:
            return progress /self.warmup
        return max((progress - 1.) / (self.warmup - 1.), 0.)

    def on_train_begin(self, trainer):
        self.t_steps = trainer.n_batches
        if self.warmup >1:
            self.warmup = self.warmup / self.t_steps
        self.t_steps = max(2, self.t_steps)  # 不能小于2
        # 防止 t_steps 不能整除 accumulation_steps
        self.t_steps = math.ceil(self.t_steps/trainer.accumulation_steps) * trainer.accumulation_steps
        # 获取param_group的初始learning rate
        for optimizer in trainer.driver.optimizers:
            for group in optimizer.param_groups:
                self.initial_lrs.append(group['lr'])

    def on_before_optimizers_step(self, trainer, optimizers):
        # 这里需要加 accumulation_steps 是防止 lr 从 0 开始
        progress = (trainer.global_forward_batches + trainer.accumulation_steps) / self.t_steps
        for optimizer in trainer.driver.optimizers:
            for lr, group in zip(self.initial_lrs, optimizer.param_groups):
                group['lr'] = lr * self.get_lr(progress)

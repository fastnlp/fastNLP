__all__ = [
    'TorchWarmupCallback'
]
import math
from typing import Union

from ..callback import Callback


class TorchWarmupCallback(Callback):
    r"""
    调整学习率的 **callback** 。

    :param warmup: 如果 ``warmup`` 为整数，则在该 step 之前，学习率根据 ``schedule`` 的策略变化; 如果 ``warmup`` 为 ``float``，
        如 0.1, 则前 10% 的 step 是按照 ``schedule`` 策略调整。
    :param schedule: 对学习率进行调整的策略：

        1. *linear* -- 前 ``warmup`` 的 step 上升到指定的学习率（从 Trainer 中 optimizer 处获取）, 在剩下的 step 中下降到 0；
        2. *constant* -- 前 ``warmup`` 的 step 上升到指定的学习率，余下的 step 保持不变。
    """
    def __init__(self, warmup:Union[int, float]=0.1, schedule:str='linear'):
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

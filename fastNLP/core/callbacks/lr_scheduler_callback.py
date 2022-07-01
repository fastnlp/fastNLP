from .callback import Callback

__all__ = [
    'LRSchedCallback'
]


class LRSchedCallback(Callback):
    """
    根据 ``step_on`` 参数在合适的时机调用 scheduler 的 step 函数。

    :param scheduler: 实现了 :meth:`step` 函数的对象；
    :param step_on: 可选 ['batch'， 'epoch'] 表示在何时调用 scheduler 的 step 函数。如果为 ``batch`` 的话在每次更新参数
        之前调用；如果为 ``epoch`` 则是在一个 epoch 运行结束后调用；
    """
    def __init__(self, scheduler, step_on:str='batch'):
        assert hasattr(scheduler, 'step') and callable(scheduler.step), "The scheduler object should have a " \
                                                                        "step function."
        self.scheduler = scheduler
        self.step_on = 0 if step_on == 'batch' else 1

    def on_after_optimizers_step(self, trainer, optimizers):
        if self.step_on == 0:
            self.scheduler.step()

    def on_train_epoch_end(self, trainer):
        if self.step_on == 1:
            self.scheduler.step()
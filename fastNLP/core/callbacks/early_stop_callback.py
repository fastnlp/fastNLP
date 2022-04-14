__all__ = [
    'EarlyStopCallback'
]

from typing import Dict, Union, Callable

from .callback import HasMonitorCallback
from fastNLP.core.utils.exceptions import EarlyStopException


class EarlyStopCallback(HasMonitorCallback):
    def __init__(self, monitor:Union[str, Callable]=None, larger_better:bool=True, patience:int=10):
        """

        :param str monitor: 监控的 metric 值。如果为 None，将尝试使用 Trainer 设置的 monitor 。也可以传入一个函数，接受参数为
            evaluation 的结果(字典类型)，返回一个 float 值作为 monitor 的结果。
        :param larger_better: monitor 的值是否是越大越好。
        :param patience: 多少次 validate 不没有提升就停止。
        """
        super(EarlyStopCallback, self).__init__(monitor=monitor, larger_better=larger_better, must_have_monitor=True)
        self.wait = 0
        self.patience = patience

    def on_validate_end(self, trainer, results):
        monitor_value = self.get_monitor_value(results)
        if monitor_value is None:
            return
        if self.is_better_monitor_value(monitor_value, keep_if_better=True):
            self.wait = 0
        else:
            self.wait += 1

    def on_fetch_data_begin(self, trainer):
        # 当是 step validate 的时候，下一步执行的就是这个， 所以在这里检查。
        if self.wait >= self.patience:
            raise EarlyStopException(f"After {self.wait} validations, no improvement for "
                                 f"metric `{self._real_monitor}`")

    def on_train_epoch_begin(self, trainer):
        # 当是 epoch validate 的时候，下一步执行的就是这个， 所以在这里检查。
        if self.wait >= self.patience:
            raise EarlyStopException(f"After {self.wait} validations, no improvement for "
                                     f"metric `{self._real_monitor}`(best value: {self.monitor_value})")

    def on_save_checkpoint(self, trainer) -> Dict:
        states = {
            'patience': self.patience,
            'wait': self.wait,
            'monitor': self.monitor,
            'monitor_value': self.monitor_value
        }
        return states

    def on_load_checkpoint(self, trainer, states):
        self.patience = states['patience']
        self.wait = states['wait']
        self.monitor = states['monitor']
        self.monitor_value = float(states['monitor_value'])

    def callback_name(self):
        return f'EarlyStopCallback#monitor-{self.monitor}#patience-{self.patience}'


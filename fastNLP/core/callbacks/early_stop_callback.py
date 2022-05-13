__all__ = [
    'EarlyStopCallback'
]

from typing import Dict, Union, Callable

from .has_monitor_callback import HasMonitorCallback
from fastNLP.core.utils.exceptions import EarlyStopException


class EarlyStopCallback(HasMonitorCallback):
    """
    用于 early stop 的 callback 。当监控的结果连续多少次没有变好边 raise 一个 EarlyStopException 。

    :param monitor: 监控的 metric 值。

        * 为 ``None``
         将尝试使用 :class:`~fastNLP.Trainer` 中设置 `monitor` 值（如果有设置）。
        * 为 ``str``
         尝试直接使用该名称从 ``evaluation`` 结果中寻找，如果在 ``evaluation`` 结果中没有找到完全一致的名称，将
         使用 最长公共字符串算法 从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor`` 。
        * 为 ``Callable``
         接受参数为 ``evaluation`` 的结果(字典类型)，返回一个 ``float`` 值作为 ``monitor`` 的结果，如果当前结果中没有相关
         的 ``monitor`` 值请返回 ``None`` 。
    :param larger_better: monitor 的值是否是越大越好。
    :param patience: 多少次 evaluate 不没有提升就停止。
    """
    def __init__(self, monitor:Union[str, Callable]=None, larger_better:bool=True, patience:int=10):
        super(EarlyStopCallback, self).__init__(monitor=monitor, larger_better=larger_better, must_have_monitor=True)
        self.wait = 0
        self.patience = patience

    def on_evaluate_end(self, trainer, results):
        monitor_value = self.get_monitor_value(results)
        if monitor_value is None:
            return
        if self.is_better_monitor_value(monitor_value, keep_if_better=True):
            self.wait = 0
        else:
            self.wait += 1

    def on_fetch_data_begin(self, trainer):
        # 当是 step evaluate 的时候，下一步执行的就是这个， 所以在这里检查。
        if self.wait >= self.patience:
            raise EarlyStopException(f"After {self.wait} validations, no improvement for "
                                     f"metric `{self._real_monitor}`(best value: {self.monitor_value})")

    def on_train_epoch_begin(self, trainer):
        # 当是 epoch evaluate 的时候，下一步执行的就是这个， 所以在这里检查。
        if self.wait >= self.patience:
            raise EarlyStopException(f"After {self.wait} validations, no improvement for "
                                     f"metric `{self._real_monitor}`(best value: {self.monitor_value})")

    def on_save_checkpoint(self, trainer) -> Dict:
        states = {
            'patience': self.patience,
            'wait': self.wait,
            'monitor_value': self.monitor_value
        }
        if not callable(self._real_monitor):
            states['_real_monitor'] = self._real_monitor
        return states

    def on_load_checkpoint(self, trainer, states):
        self.patience = states['patience']
        self.wait = states['wait']
        self.monitor_value = float(states['monitor_value'])
        if '_real_monitor' in states:
            self._real_monitor = states['_real_monitor']

    @property
    def callback_name(self):
        return f'EarlyStopCallback#monitor-{self.monitor_name}#patience-{self.patience}'


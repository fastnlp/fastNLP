__all__ = [
    'HasMonitorCallback',
    'ExecuteOnceBetterMonitor',
    'MonitorUtility'
]

from typing import Dict, Union, Any
from abc import ABC
import functools

from fastNLP.core.utils import apply_to_collection
from fastNLP.core.callbacks import Callback
from fastNLP.core.callbacks.utils import _get_monitor_value
from fastNLP.core.log import logger
from fastNLP.core.utils.utils import _check_valid_parameters_number


class CanItemDataType(ABC):
    """
    检测可以进行传输的对象。

    """

    @classmethod
    def __subclasshook__(cls, subclass: Any) -> Union[bool, Any]:
        if cls is CanItemDataType:
            item = getattr(subclass, 'item', None)
            return callable(item)
        return NotImplemented


class MonitorUtility:
    """
    计算 monitor 的相关函数

    """
    def __init__(self, monitor, larger_better):
        self.set_monitor(monitor, larger_better)

    def set_monitor(self, monitor, larger_better):
        if callable(monitor):  # 检查是否能够接受一个参数
            _check_valid_parameters_number(monitor, expected_params=['results'], fn_name='monitor')
            self.monitor = monitor
        else:
            self.monitor = str(monitor) if monitor is not None else None
        if self.monitor is not None:
            self.larger_better = bool(larger_better)
        if larger_better:
            self.monitor_value = float('-inf')
        else:
            self.monitor_value = float('inf')
        self._real_monitor = self.monitor

    def itemize_results(self, results):
        """
        将结果中有 .item() 方法的都调用一下，使得可以结果可以保存

        :param results:
        :return:
        """
        return apply_to_collection(results, dtype=CanItemDataType, function=lambda x: x.item())

    def get_monitor_value(self, results:Dict)->Union[float, None]:
        """
        获取 monitor 的值，如果 monitor 没有直接找到，会尝试使用匹配的方式寻找，并把匹配到的设置到 self._real_monitor 属性上。

        :param results:
        :return: 如果为 None ，表明此次没有找到合适的monitor
        """
        if len(results) == 0 or self.monitor is None:
            return None
        # 保证所有的 tensor 都被转换为了 python 特定的类型
        results = self.itemize_results(results)
        use_monitor, monitor_value = _get_monitor_value(monitor=self.monitor,
                                                        real_monitor=self._real_monitor,
                                                        res=results)
        if monitor_value is None:
            return monitor_value
        # 第一次运行
        if isinstance(self.monitor, str) and self._real_monitor == self.monitor and use_monitor != self.monitor:
            logger.rank_zero_warning(f"We can not find `{self.monitor}` in the evaluation result (with keys as "
                                     f"{list(results.keys())}), we use the `{use_monitor}` as the monitor.", once=True)
        # 检测到此次和上次不同。
        elif isinstance(self.monitor, str) and self._real_monitor != self.monitor and use_monitor != self._real_monitor:
            logger.rank_zero_warning(f"Change of monitor detected for `{self.__class__.__name__}`. "
                           f"The expected monitor is:`{self.monitor}`, last used monitor is:"
                           f"`{self._real_monitor}` and current monitor is:`{use_monitor}`. Please consider using a "
                           f"customized monitor function when the evaluation results are varying between validation.")

        self._real_monitor = use_monitor
        return monitor_value

    def is_better_monitor_value(self, monitor_value: float, keep_if_better=True):
        """
        检测 monitor_value 是否是更好的

        :param monitor_value: 待检查的 monitor_value 。如果为 None ，返回 False
        :param keep_if_better: 如果传入的 monitor_value 值更好，则将其保存下来。
        :return:
        """
        if monitor_value is None:
            return False
        better = self.is_former_monitor_value_better(monitor_value, self.monitor_value)
        if keep_if_better and better:
            self.monitor_value = monitor_value
        return better

    def is_better_results(self, results, keep_if_better=True):
        """
        检测给定的 results 是否比上一次更好，如果本次 results 中没有找到相关的monitor 返回 False。

        :param results: on_valid_ends() 接口中传入的 evaluation 结果。
        :param keep_if_better: 当返回为 True 时，是否保存到 self.monitor_value 中。
        :return:
        """
        monitor_value = self.get_monitor_value(results)
        if monitor_value is None:
            return False
        return self.is_better_monitor_value(monitor_value, keep_if_better=keep_if_better)

    def is_former_monitor_value_better(self, monitor_value1, monitor_value2):
        """
        传入的两个值中，是否monitor_value1的结果更好。

        :param monitor_value1:
        :param monitor_value2:
        :return:
        """
        if monitor_value1 is None and monitor_value2 is None:
            return True
        if monitor_value1 is None:
            return False
        if monitor_value2 is None:
            return True
        better = False
        if (self.larger_better and monitor_value1 > monitor_value2) or \
                (not self.larger_better and monitor_value1 < monitor_value2):
            better = True
        return better

    @property
    def monitor_name(self):
        """
        返回 monitor 的名字，如果 monitor 是个 callable 的函数，则返回该函数的名称。

        :return:
        """
        if callable(self.monitor):
            try:
                monitor = self.monitor
                while isinstance(monitor, functools.partial):
                    monitor = monitor.func
                monitor_name = monitor.__qualname__
            except:
                monitor_name = self.monitor.__name__
        elif self.monitor is None:
            return None
        else:
            # 这里是能是monitor，而不能是real_monitor，因为用户再次运行的时候real_monitor被初始化为monitor了
            monitor_name = str(self.monitor)
        return monitor_name


class HasMonitorCallback(MonitorUtility, Callback):
    def __init__(self, monitor, larger_better, must_have_monitor=False):
        """
        该 callback 不直接进行使用，作为其它相关 callback 的父类使用，如果 callback 有使用 monitor 可以继承该函数里面实现了
        （1）判断monitor合法性；（2）在需要时， 根据trainer的monitor设置自己的monitor名称。

        :param monitor: 监控的 metric 值。如果在 evaluation 结果中没有找到完全一致的名称，将使用 最短公共字符串算法 找到最匹配
                的那个作为 monitor 。如果为 None，将尝试使用 Trainer 设置的 monitor 。也可以传入一个函数，接受参数为 evaluation 的结
                果(字典类型)，返回一个 float 值作为 monitor 的结果，如果当前结果中没有相关的 monitor 值请返回 None 。
        :param larger_better: monitor 是否时越大越好
        :param must_have_monitor: 这个 callback 是否必须有 monitor 设置。如果设置为 True ，且没检测到设置 monitor 会报错。
        """
        super().__init__(monitor, larger_better)
        self.must_have_monitor = must_have_monitor

    def on_after_trainer_initialized(self, trainer, driver):
        """
        如果本身的 monitor 没有设置，则根据 Trainer 中的 monitor 设置 monitor 。
        同时对于必须要有 monitor 设置的 callback ，该函数会进行检查。

        :param trainer:
        :param driver:
        :return:
        """
        if self.monitor is None and trainer.monitor is not None:
            self.set_monitor(monitor=trainer.monitor, larger_better=trainer.larger_better)
        if self.must_have_monitor and self.monitor is None:
            raise RuntimeError(f"No `monitor` is set for {self.__class__.__name__}. "
                               f"You can set it in the initialization or through Trainer.")
        if self.must_have_monitor and self.monitor is not None and trainer.evaluator is None:
            raise RuntimeError(f"No `evaluate_dataloaders` is set for Trainer. But Callback: {self.__class__.__name__}"
                               f" need to watch the monitor:`{self.monitor_name}`.")

    def on_sanity_check_end(self, trainer, sanity_check_res):
        # 主要核对一下 monitor 是否存在。
        if self.monitor is not None:
            self.get_monitor_value(results=sanity_check_res)


class ExecuteOnceBetterMonitor(HasMonitorCallback):
    def __init__(self, monitor, larger_better, execute_fn):
        """
        当监控的 monitor 结果更好的时候，调用 execute_fn 函数。

        :param monitor: 监控的 metric 值。如果在 evaluation 结果中没有找到完全一致的名称，将使用 最短公共字符串算法 找到最匹配
                的那个作为 monitor 。如果为 None，将尝试使用 Trainer 设置的 monitor 。也可以传入一个函数，接受参数为 evaluation 的结
                果(字典类型)，返回一个 float 值作为 monitor 的结果，如果当前结果中没有相关的 monitor 值请返回 None 。
        :param larger_better: monitor 是否时越大越好
        :param execute_fn: 一个可执行的函数，不接受任何参数，不反回值。在 monitor 取得更好结果的时候会调用。
        """
        super().__init__(monitor, larger_better, must_have_monitor=True)
        _check_valid_parameters_number(execute_fn, expected_params=[], fn_name='execute_fn')
        self.execute_fn = execute_fn

    def on_evaluate_end(self, trainer, results):
        if self.is_better_results(results):
            self.execute_fn()
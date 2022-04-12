from typing import Union, Callable, Dict, Optional, Any
from abc import ABC

__all__ = [
    'Callback',
]

from .callback_events import Events, EventsList, Filter
from .utils import _get_monitor_value
from fastNLP.core.callbacks.callback_events import _SingleEventState
from fastNLP.core.log import logger
from fastNLP.core.utils import apply_to_collection


class Callback:
    r"""
    实际使用的 callback 类，不管是我们 fastNLP 默认提供的一些 callback 类，还是用户自己定制的 callback 类，都应该继承该基类；
    """

    def on_after_trainer_initialized(self, trainer, driver):
        r"""
        在 `Trainer` 初始化后会被触发；
        """
        pass

    def on_sanity_check_begin(self, trainer):
        r"""
        在 '预跑'检测 开始前会被触发；
        """
        pass

    def on_sanity_check_end(self, trainer, sanity_check_res):
        r"""
        在 '预跑'检测 开始后会被触发；
        """
        pass

    def on_train_begin(self, trainer):
        r"""
        在训练开始前会被触发；
        """
        pass

    def on_train_end(self, trainer):
        r"""
        在训练完成后会被触发；
        """
        pass

    def on_train_epoch_begin(self, trainer):
        r"""
        在训练过程中的每一个 epoch 开始前会被触发；
        """
        pass

    def on_train_epoch_end(self, trainer):
        r"""
        在训练过程中的每一个 epoch 完成后会被触发；
        """
        pass

    def on_fetch_data_begin(self, trainer):
        r"""
        在训练过程中拿到当前的具体的一个 batch 前会被触发；
        """
        pass

    def on_fetch_data_end(self, trainer):
        r"""
        在训练过程中拿到当前的具体的一个 batch 后会被触发；
        """
        pass

    def on_train_batch_begin(self, trainer, batch, indices):
        r"""
        在训练过程中开始具体的一个 batch 前会被触发；

        :param trainer: `fastNLP.Trainer`
        :param batch: 当前正在运行的一个 batch；
        :param indices: 当前的 batch 在一个 epoch 中的位置，用于用户方便地通过该 callback 函数定位具体的数据；
        """
        pass

    def on_train_batch_end(self, trainer):
        pass

    def on_exception(self, trainer, exception):
        pass

    def on_save_model(self, trainer):
        pass

    def on_load_model(self, trainer):
        pass

    def on_save_checkpoint(self, trainer) -> Dict:
        """
        当确定前后两个 callback 是一样的（callback_name 相同，意味着它们所起的职能相同）时，它们在该函数中则应当保存使该 callback 正常
        工作的状态；而不应该让该函数去判断两个 callback 是否一样；
        """
        pass

    def on_load_checkpoint(self, trainer, states: Optional[Dict]):
        r"""
        如果一个 callback 在断点重训前没有保存状态，或者其 `callback_name` 与其余的 callback 重名时，`states` 为 None；
        """
        pass

    def on_before_backward(self, trainer, outputs):
        pass

    def on_after_backward(self, trainer):
        pass

    def on_before_optimizer_step(self, trainer, optimizers):
        pass

    def on_before_zero_grad(self, trainer, optimizers):
        pass

    def on_validate_begin(self, trainer):
        pass

    def on_validate_end(self, trainer, results):
        pass

    @property
    def callback_name(self):
        return self.__class__.__name__


class _CallbackWrapper(Callback):
    """
    对于用户使用函数修饰器加入的 callback 函数，使用该 _CallbackWrapper 类为其进行定制，这一个类只保留用户的
     这一个 callback 函数；
    """
    def __init__(self, event: Union[Events, EventsList], fn: Callable):
        r"""
        :param event: 具体的 callback 时机，例如 'on_train_begin' 等；可以多个时机，此时 `event` 的 type 应当为 'EventsList'；
        :param fn: 用户定制的 callback 函数；
        """

        self.fn = fn
        if isinstance(event, EventsList):
            for each_event in event:
                _filter = Filter(each_event.every, each_event.once, each_event.filter_fn)
                setattr(self, each_event.value, _filter(fn))
        elif isinstance(event, _SingleEventState):
            _filter = Filter(event.every, event.once, event.filter_fn)
            setattr(self, event.value, _filter(fn))

    @property
    def callback_name(self):
        return self.fn.__name__


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


class HasMonitorCallback(Callback):
    def __init__(self, monitor, larger_better, must_have_monitor=False):
        self.set_monitor(monitor, larger_better)
        self.must_have_moinitor = must_have_monitor

    def set_monitor(self, monitor, larger_better):
        self.monitor = str(monitor) if monitor is not None else None
        self.larger_better = bool(larger_better)
        if larger_better:
            self.monitor_value = float('-inf')
        else:
            self.monitor_value = float('inf')
        self._real_monitor = self.monitor

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
        if self.must_have_moinitor and self.monitor is None:
            raise RuntimeError(f"No `monitor` is set for {self.__class__.__name__}. "
                               f"You can set it in the initialization or through Trainer.")

    def get_monitor_value(self, results:Dict)->float:
        """
        获取 monitor 的值，如果 monitor 没有直接找到，会尝试使用匹配的方式寻找，并把匹配到的设置到 self._real_monitor 属性上。

        :param results:
        :return:
        """
        if len(results)==0:
            return 0
        # 保证所有的 tensor 都被转换为了 python 特定的类型
        results = apply_to_collection(results, dtype=CanItemDataType, function=lambda x: x.item())
        use_monitor, monitor_value = _get_monitor_value(monitor=self.monitor,
                                                        real_monitor=self._real_monitor,
                                                        res=results)
        if self._real_monitor != use_monitor:  # 发生了替换需要打印
            logger.warning(
                f"We can not find `{self.monitor}` in the evaluation result (with keys as {list(results.keys())}), "
                f"we use the `{use_monitor}` as the monitor for {self.__class__.__name__}.")
        self._real_monitor = use_monitor
        return monitor_value

    def is_better_monitor_value(self, monitor_value: float, keep_if_better=True):
        """
        检测 monitor_value 是否是更好的

        :param monitor_value:
        :param keep_if_better: 如果传入的 monitor_value 值更好，则将其保存下来。
        :return:
        """
        better = False
        if (self.larger_better and monitor_value > self.monitor_value) or \
            (not self.larger_better and monitor_value < self.monitor_value):
            better = True
            if keep_if_better:
                self.monitor_value = monitor_value
        return better
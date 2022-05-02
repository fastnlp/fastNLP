from enum import Enum, unique
from typing import Union, Optional, List, Iterator, Callable, Tuple, Dict
from types import DynamicClassAttribute
from functools import wraps


__all__ = [
    'Events',
    'EventsList',
    'Filter'
]


class _SingleEventState:
    every: Optional[int]
    once: Optional[int]

    def __init__(self, value: str, every: Optional[int] = None, once: Optional[int] = None,
                 filter_fn: Optional[Callable] = None, name: Optional[str] = None):

        # 具体的检测参数对错的逻辑放在具体的 Filter 里；
        if every is None and once is None and filter_fn is None:
            self.every = 1
            self.once = None
            self.filter_fn = None
        else:
            self.every = every
            self.once = once
            self.filter_fn = filter_fn

        if not hasattr(self, "_value_"):
            self._value_ = value

        if not hasattr(self, "_name_") and name is not None:
            self._name_ = name

    # copied to be compatible to enum
    @DynamicClassAttribute
    def name(self) -> str:
        """The name of the Enum member."""
        return self._name_

    @DynamicClassAttribute
    def value(self) -> str:
        """The value of the Enum member."""
        return self._value_

    def __call__(self, every: Optional[int] = None, once: Optional[int] = None, filter_fn: Optional[Callable] = None):
        return _SingleEventState(self.value, every, once, filter_fn, self.name)

    def __str__(self):
        return "<event={0}, every={1}, once={2}, filter fn is None:{3}>".format(self.name, self.every, self.once,
                                                                                self.filter_fn)

    def __eq__(self, other) -> bool:
        if isinstance(other, _SingleEventState):
            return self.name == other.name
        elif isinstance(other, str):
            return self.name == other
        else:
            raise NotImplemented

    def __hash__(self):
        return hash(self._name_)

    def __or__(self, other) -> "EventsList":
        return EventsList() | self | other


class EventEnum(_SingleEventState, Enum):
    pass

@unique
class Events(EventEnum):
    on_after_trainer_initialized = "on_after_trainer_initialized"
    on_sanity_check_begin = "on_sanity_check_begin"
    on_sanity_check_end = "on_sanity_check_end"
    on_train_begin = "on_train_begin"
    on_train_end = "on_train_end"
    on_train_epoch_begin = "on_train_epoch_begin"
    on_train_epoch_end = "on_train_epoch_end"
    on_fetch_data_begin = "on_fetch_data_begin"
    on_fetch_data_end = "on_fetch_data_end"
    on_train_batch_begin = "on_train_batch_begin"
    on_train_batch_end = "on_train_batch_end"
    on_exception = "on_exception"
    on_save_model = "on_save_model"
    on_load_model = "on_load_model"
    on_save_checkpoint = "on_save_checkpoint"
    on_load_checkpoint = "on_load_checkpoint"
    on_before_backward = "on_before_backward"
    on_after_backward = "on_after_backward"
    on_before_optimizers_step = "on_before_optimizers_step"
    on_after_optimizers_step = "on_after_optimizers_step"
    on_before_zero_grad = "on_before_zero_grad"
    on_after_zero_grad = "on_after_zero_grad"
    on_evaluate_begin = "on_evaluate_begin"
    on_evaluate_end = "on_evaluate_end"


class EventsList:
    """Collection of events stacked by operator `__or__`.
    """

    def __init__(self) -> None:
        self._events = []  # type: List[Union[Events, _SingleEventState]]

    def _append(self, event: Union[Events, _SingleEventState]) -> None:
        if not isinstance(event, (Events, _SingleEventState)):
            raise TypeError(f"Argument event should be Events or CallableEventWithFilter, got: {type(event)}")
        self._events.append(event)

    def __getitem__(self, item: int) -> Union[Events, _SingleEventState]:
        return self._events[item]

    def __iter__(self) -> Iterator[Union[Events, _SingleEventState]]:
        return iter(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def __or__(self, other: Union[Events, _SingleEventState]) -> "EventsList":
        self._append(event=other)
        return self


class Filter:
    def __init__(self, every: Optional[int] = None, once: Optional[int] = None, filter_fn: Optional[Callable] = None):
        r"""
        通过该 `Filter` 作为函数修饰器来控制一个函数的实际的运行频率；

        :param every: 表示一个函数隔多少次运行一次；
        :param once: 表示一个函数只在第多少次时运行一次；
        :param filter_fn: 用户定制的频率控制函数；注意该函数内部的频率判断应当是无状态的，除了参数 `self.num_called` 和
         `self.num_executed` 外，因为我们会在预跑后重置这两个参数的状态；
        """
        if (every is None) and (once is None) and (filter_fn is None):
            raise ValueError("If you mean your decorated function should be called every time, you do not need this filter.")

        if not ((every is not None) ^ (once is not None) ^ (filter_fn is not None)):
            raise ValueError("These three values should be only set one.")

        if (filter_fn is not None) and not callable(filter_fn):
            raise TypeError("Argument event_filter should be a callable")

        if (every is not None) and not (isinstance(every, int) and every > 0):
            raise ValueError("Argument every should be integer and greater than zero")

        if (once is not None) and not (isinstance(once, int) and once > 0):
            raise ValueError("Argument once should be integer and positive")

        # 设置变量，包括全局变量；
        self.num_called = 0
        self.num_executed = 0

        if every is not None:
            self._every = every
            self._filter = self.every_filter
        elif once is not None:
            self._once = once
            self._filter = self.once_filter
        else:
            self._filter = filter_fn

    def __call__(self, fn: Callable):

        @wraps(fn)
        def wrapper(*args, **kwargs) -> Callable:
            self.num_called += 1

            # 因为我们的 callback 函数的输入是固定的，而且我们能够保证第一个参数一定是 trainer；
            trainer = args[0]
            if self._filter(self, trainer):
                self.num_executed += 1
                return fn(*args, **kwargs)

        wrapper.__fastNLP_filter__ = self
        return wrapper

    def every_filter(self, *args):
        return self.num_called % self._every == 0

    def once_filter(self, *args):
        return self.num_called == self._once

    def state_dict(self) -> Dict:
        r"""
        通过该函数来保存该 `Filter` 的状态；
        """
        return {"num_called": self.num_called, "num_executed": self.num_executed}

    def load_state_dict(self, state: Dict):
        r"""
        通过该函数来加载 `Filter` 的状态；

        :param state: 通过 `Filter.state_dict` 函数保存的状态元组；
        """
        self.num_called = state["num_called"]
        self.num_executed = state["num_executed"]








__all__ = [
    'Element'
]

import os
import functools

from .backend import Backend, AutoBackend
from fastNLP.core.log import logger
from .utils import AggregateMethodError
from fastNLP.envs.env import FASTNLP_GLOBAL_RANK


def _wrap_cal_value(func):
    @functools.wraps(func)
    def _wrap_cal(*args, **kwargs):
        self = args[0]
        value = func(*args, **kwargs)
        value = self.backend.get_scalar(value)
        return value

    return _wrap_cal


class Element:
    def __init__(self, name, value: float, aggregate_method, backend: Backend):
        """
        保存 Metric 中计算的元素值的对象

        :param name: 名称
        :param value: 元素的值
        :param aggregate_method: 聚合的方法， 目前支持 ``['sum', 'mean', 'max', 'mix']``:

            * method 为 ``'sum'`` 时， 会将多张卡上聚合结果在维度为 `0` 上 累加起来。
            * method 为 ``'mean'`` 时，会将多张卡上聚合结果在维度为 `0` 上取平均值。
            * method 为 ``'max'`` 时，会将多张卡上聚合结果在维度为 `0` 上取最大值。
            * method 为 ``'mix'`` 时，会将多张卡上聚合结果在维度为 `0` 上取最小值。

        :param backend: 使用的 backend 。Element 的类型会根据 backend 进行实际的初始化。例如 backend 为 torch 则该对象为
            Torch.tensor ； 如果backend 为 paddle 则该对象为 paddle.tensor ；如果 backend 为 jittor , 则该对象为 jittor.Var 。
            一般情况下直接默认为 auto 就行了，fastNLP 会根据实际调用 Metric.update() 函数时传入的参数进行合理的初始化，例如当传入
            的参数中只包含 torch.Tensor 这一种 tensor 时（可以有其它非 tensor 类型的输入）则认为 backend 为 torch ；只包含
            jittor.Var 则认为 backend 这一种 tensor 时（可以有其它非 tensor 类型的输入）则认为 backend 为 jittor 。如果没有检测
            到任何一种 tensor ，就默认使用 float 类型作为 element 。

        """
        self.name = name
        self.init_value = value
        self.aggregate_method = aggregate_method
        if backend == 'auto':
            raise RuntimeError(f"You have to specify the backend for Element:{self.name}.")
        elif isinstance(backend, AutoBackend):
            self.backend = backend
        else:
            self.backend = AutoBackend(backend)

        if self.backend.is_specified():
            value = self.backend.create_tensor(self.init_value)
        else:
            value = None
        self._value = value
        self.device = None

    def aggregate(self):
        """
        自动 aggregate 对应的元素

        """
        self._check_value_initialized()
        if self.aggregate_method is None:  # 如果没有 aggregate 则不进行聚合。
            return
        try:
            self._value = self.backend.aggregate(self._value, self.aggregate_method)
        except AggregateMethodError as e:
            msg = 'If you see this message, please report a bug.'
            if self.name and e.should_have_aggregate_method:
                msg = f"Element:{self.name} has no specified `aggregate_method`."
            elif self.name and not e.should_have_aggregate_method:
                msg = f"Element:{self.name}'s backend:{self.backend.__class__.__name__} does not support " \
                      f'aggregate_method:{self.aggregate_method}.'
            if e.only_warn:
                if int(os.environ.get(FASTNLP_GLOBAL_RANK, 0)) == 0:
                    logger.warning(msg)
                self._value = self.backend.aggregate(self._value, method=None)
            else:
                raise RuntimeError(msg)

    def reset(self):
        """
        重置 value
        """
        if self.backend.is_specified():
            self._value = self.backend.fill_value(self._value, self.init_value)

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._check_value_initialized()
        self._value = value

    @value.getter
    def value(self):
        self._check_value_initialized()
        return self._value

    def get_scalar(self) -> float:
        """
        获取元素的 scalar 值

        """
        self._check_value_initialized()
        return self.backend.get_scalar(self._value)

    def fill_value(self, value):
        """
        对元素进行 fill_value， 会执行队友 backend 的 fill_value 方法

        """
        self._check_value_initialized()
        self._value = self.backend.fill_value(self._value, value)

    def to(self, device):
        """
        将元素移到某个设备上

        :param device: 设备名， 一般为 ``"cpu"``, ``"cuda:0"`` 等
        """
        # device这里如何处理呢？
        if self._value is not None:
            self._value = self.backend.move_tensor_to_device(self._value, device)
        self.device = device

    def _check_value_initialized(self):
        """
        检查 Element 的 value 是否初始化了

        """
        if self._value is None:
            assert self.backend.is_specified(), f"Backend is not specified, please specify backend in the Metric " \
                                                f"initialization."
            self._value = self.backend.create_tensor(self.init_value)
            if self.device is not None:
                self.to(device=self.device)

    def _check_value_when_call(self):
        if self.value is None:
            prefix = f'Element:`{self.name}`'
            raise RuntimeError(prefix + " is not initialized. Please either specify backend when creating this "
                                        "element, or use it after it being used by the `Metric.update()` method.")

    @_wrap_cal_value
    def __add__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value + other

    @_wrap_cal_value
    def __radd__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value + other

    @_wrap_cal_value
    def __sub__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value - other

    @_wrap_cal_value
    def __rsub__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value - other

    @_wrap_cal_value
    def __mul__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value * other

    @_wrap_cal_value
    def __imul__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value * other

    @_wrap_cal_value
    def __floordiv__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value // other

    @_wrap_cal_value
    def __rfloordiv__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value // other

    @_wrap_cal_value
    def __truediv__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value / other

    @_wrap_cal_value
    def __rtruediv__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value / other

    @_wrap_cal_value
    def __mod__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value % other

    @_wrap_cal_value
    def __rmod__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value % other

    @_wrap_cal_value
    def __pow__(self, other, modulo=None):
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        if modulo is None:
            return self.value ** other
        else:
            return pow(self.value, other, modulo)

    @_wrap_cal_value
    def __rpow__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value ** other

    @_wrap_cal_value
    def __lt__(self, other) -> bool:
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value < other

    @_wrap_cal_value
    def __le__(self, other) -> bool:
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value <= other

    @_wrap_cal_value
    def __eq__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value == other

    @_wrap_cal_value
    def __ne__(self, other) -> bool:
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value != other

    @_wrap_cal_value
    def __ge__(self, other) -> bool:
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value >= other

    @_wrap_cal_value
    def __gt__(self, other) -> bool:
        self._check_value_when_call()
        if isinstance(other, Element):
            other = other.value
        return self.value > other

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    def __getattr__(self, item):
        """
        为FDataLoader提供dataset的方法和属性，实现该方法后，用户可以在FDataLoader实例化后使用apply等dataset的方法
        :param item:
        :return:
        """
        try:
            if self._value is None:
                prefix = f'Element:`{self.name}`'
                raise RuntimeError(prefix + " is not initialized. Please either specify backend when creating this "
                                            "element, or use it after it being used by the `Metric.update()` method.")
            return getattr(self._value, item)
        except AttributeError as e:
            logger.error(f"Element:{self.name} has no `{item}` attribute.")
            raise e

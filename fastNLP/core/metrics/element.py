__all__ = [
    'Element'
]

import os

from .backend import Backend, AutoBackend
from fastNLP.core.log import logger
from .utils import AggregateMethodError
from fastNLP.envs.env import FASTNLP_GLOBAL_RANK


class Element:
    def __init__(self, value: float, aggregate_method, backend: Backend, name=None):
        self.init_value = value
        self.aggregate_method = aggregate_method
        self.name = name
        if backend == 'auto':
            raise RuntimeError("You have to specify the backend.")
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
        自动aggregate对应的元素

        """
        self._check_value_initialized()
        try:
            self._value = self.backend.aggregate(self._value, self.aggregate_method)
        except AggregateMethodError as e:
            msg = 'If you see this message, please report a bug.'
            if self.name and e.should_have_aggregate_method:
                msg = f"Element:{self.name} has no specified `aggregate_method`."
            elif e.should_have_aggregate_method:
                msg = "Element has no specified `aggregate_method`."
            elif self.name and not e.should_have_aggregate_method:
                msg = f"Element:{self.name}'s backend:{self.backend.__class__.__name__} does not support " \
                      f'aggregate_method:{self.aggregate_method}.'
            elif not e.should_have_aggregate_method:
                msg = f"Element's backend:{self.backend.__class__.__name__} does not support " \
                      f'aggregate_method:{self.aggregate_method}.'
            if e.only_warn:
                if int(os.environ.get(FASTNLP_GLOBAL_RANK, 0)) == 0:
                    logger.warning(msg)
                self._value = self.backend.aggregate(self._value, method=None)
            else:
                raise RuntimeError(msg)

    def reset(self):
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
        self._check_value_initialized()
        return self.backend.get_scalar(self._value)

    def fill_value(self, value):
        self._value = self.backend.fill_value(self._value, value)

    def to(self, device):
        # device这里如何处理呢？
        if self._value is not None:
            self._value = self.backend.move_tensor_to_device(self._value, device)
        self.device = device

    def _check_value_initialized(self):
        if self._value is None:
            assert self.backend.is_specified(), f"Backend is not specified, please specify backend in the Metric " \
                                                f"initialization."
            self._value = self.backend.create_tensor(self.init_value)
            if self.device is not None:
                self.to(device=self.device)

    def _check_value_when_call(self):
        if self.value is None:
            prefix = f'Element:`{self.name}`' if self.name else 'Element'
            raise RuntimeError(prefix + " is not initialized. Please either specify backend when creating this "
                                        "element, or use it after it being used by the `Metric.compute()` method.")

    def __add__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            self.value += other.value
        else:
            self.value += other
        return self

    def __radd__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            self.value += other.value
        else:
            self.value += other
        return self

    def __sub__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            self.value -= other.value
        else:
            self.value -= other
        return self

    def __rsub__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            self.value -= other.value
        else:
            self.value -= other
        return self

    def __mul__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            self.value *= other.value
        else:
            self.value *= other
        return self

    def __imul__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            self.value *= other.value
        else:
            self.value *= other
        return self

    def __floordiv__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            self.value //= other.value
        else:
            self.value //= other
        return self

    def __rfloordiv__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            self.value //= other.value
        else:
            self.value //= other
        return self

    def __truediv__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            self.value /= other.value
        else:
            self.value /= other
        return self

    def __rtruediv__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            self.value /= other.value
        else:
            self.value /= other
        return self

    def __mod__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            self.value %= other.value
        else:
            self.value %= other
        return self

    def __rmod__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            self.value /= other.value
        else:
            self.value /= other
        return self

    def __pow__(self, other, modulo=None):
        self._check_value_when_call()
        if modulo is None:
            if isinstance(other, Element):
                self.value **= other.value
            else:
                self.value **= other
        else:
            if isinstance(other, Element):
                self.value = pow(self.value, other.value, modulo)
            else:
                self.value = pow(self.value, other, modulo)
        return self

    def __rpow__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            self.value **= other.value
        else:
            self.value **= other
        return self

    def __lt__(self, other) -> bool:
        self._check_value_when_call()
        if isinstance(other, Element):
            return self.value < other.value
        else:
            return self.value < other

    def __le__(self, other) -> bool:
        self._check_value_when_call()
        if isinstance(other, Element):
            return self.value <= other.value
        else:
            return self.value <= other

    def __eq__(self, other):
        self._check_value_when_call()
        if isinstance(other, Element):
            return self.value == other.value
        else:
            return self.value == other

    def __ne__(self, other) -> bool:
        self._check_value_when_call()
        if isinstance(other, Element):
            return self.value != other.value
        else:
            return self.value != other

    def __ge__(self, other) -> bool:
        self._check_value_when_call()
        if isinstance(other, Element):
            return self.value >= other.value
        else:
            return self.value >= other

    def __gt__(self, other) -> bool:
        self._check_value_when_call()
        if isinstance(other, Element):
            return self.value > other.value
        else:
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
                prefix = f'Element:`{self.name}`' if self.name else 'Element'
                raise RuntimeError(prefix + " is not initialized. Please either specify backend when creating this "
                                            "element, or use it after it being used by the `Metric.compute()` method.")
            return getattr(self._value, item)
        except AttributeError as e:
            raise e

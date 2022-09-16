__all__ = [
    'Metric'
]

from abc import abstractmethod

from typing import Union, List
import functools
from contextlib import contextmanager
import numpy as np

from fastNLP.core.metrics.backend import Backend, AutoBackend
from fastNLP.core.metrics.element import Element


class Metric:
    """
    **fastNLP** 中 :class:`Metric` 的基类，自定义 :class:`Metric` 时，请继承该对象。使用该对象，将有助于减少在分布式状态下的 Metric 计算。

    :param backend: 目前支持五种类型的 backend, ``['torch', 'paddle', 'jittor', 'oneflow', 'auto']``。其中 ``'auto'`` 表示根据实际调用 :meth:`update`
        函数时传入的参数决定具体的 backend ，大部分情况下直接使用 ``'auto'`` 即可。
    :param aggregate_when_get_metric: 在计算 metric 的时候是否自动将各个进程上的相同的 element 的数字聚合后再得到 metric，
        当 backend 不支持分布式时，该参数无意义。如果为 ``None`` ，将在 :class:`~fastNLP.core.controllers.Evaluator` 中根据
        sampler 是否使用分布式进行自动设置。
    """
    def __init__(self, backend: Union[str, Backend, None] = 'auto', aggregate_when_get_metric: bool = None):
        self.backend = AutoBackend(backend)
        self._updated = False
        self.get_metric = self._sync_get_metric(self.get_metric)
        self.update = self._wrap_update(self.update)
        self.reset = self._wrap_auto_reset_elements(self.reset)
        self.aggregate_when_get_metric = aggregate_when_get_metric
        self._cannot_change_element = False
        self._elements = {}

    @property
    def elements(self) -> dict:
        return self._elements

    def register_element(self, name, value: float = 0, aggregate_method=None, backend='auto') -> Element:
        """
        注册一个 element 对象，注册之后便可以通过在 Metric 中直接通过 ``self.{name}`` 进行调用，可以认为该对象即为对应 backend 的
        tensor 直接进行加减乘除计算即可。

        .. warning::

            如果想使得该 metric 可自动扩展到多卡的情况，请一定申明 ``aggregate_method`` 。

        :param name: 当前 element 的名字，注册后，在 Metric 中可以通过 ``self.{name}`` 访问该变量。
        :param value: 初始化的值。在调用 :meth:`Metric.reset` 方法时也将自动设置为该值
        :param aggregate_method: 如何聚合多卡上的结果，如果为单卡执行，该值无意义。如果设置为 None 则表示该 element 不进行聚合。
        :param backend: 使用的 backend 。Element 的类型会根据 ``backend`` 进行实际的初始化。例如 ``backend`` 为 ``'torch'`` 则该对象为
            :class:`torch.Tensor` ； 如果 ``'backend'`` 为 ``'paddle'`` 则该对象为 :class:`paddle.Tensor` ；如果 ``backend`` 为
            ``'jittor'`` , 则该对象为 :class:`jittor.Var` 。一般情况下直接默认为 ``'auto'`` 就行了， **fastNLP** 会根据实际调用 :meth`Metric.update`
            函数时传入的参数进行合理的初始化，例如当传入的参数中只包含 :class:`torch.Tensor` 这一种 tensor 时（可以有其它非 tensor 类型的输入）
            则认为 ``backend`` 为 ``'torch'`` ；只包含 :class:`jittor.Var` 这一种 tensor 时（可以有其它非 tensor 类型的输入）则认为 ``backend``
            为 ``'jittor'`` 。如果没有检测到任何一种 tensor ，就默认使用 :class:`float` 类型作为 element 。
        :return: 注册的 Element 对象
        """
        if backend == 'auto':
            backend = self.backend
        else:
            backend = AutoBackend(backend)

        assert name is not None and name not in self.elements

        element = Element(name=name, value=value, aggregate_method=aggregate_method, backend=backend)
        self.elements[name] = element
        setattr(self, name, element)
        return element

    def reset(self):
        """
        在对每个 ``evaluate_dataloaders`` 遍历进行验证之前，:meth:`reset` 函数会被调用来重置每个非 element 对象；
        如果有非 element 的对象需要重置的时候，在本方法中写下非 element 的重置方式。注册的 element 对象则会自动 reset 为初始值。
        """
        pass

    def _wrap_auto_reset_elements(self, reset):
        @functools.wraps(reset)
        def _wrap_reset(*args, **kwargs):
            self._updated = False
            for ele in self.elements.values():
                ele.reset()
            reset(*args, **kwargs)

        return _wrap_reset

    def _sync_get_metric(self, get_metric):
        @functools.wraps(get_metric)
        def _wrap_get_metric(*args, **kwargs):
            assert self._updated, f"You have to call `{self.__class__.__name__}'s update() function before calling " \
                                  f"get_metric()."
            with self.sync(recover=True, aggregate=self.aggregate_when_get_metric):
                results = get_metric(*args, **kwargs)
            return results

        return _wrap_get_metric

    def __setattr__(self, key, value):
        if getattr(self, '_cannot_change_element', False):
            if key in self.elements and isinstance(value, (float, int, bool)):
                self.elements[key].fill_value(value)
                return
            elif key in self.elements:
                raise TypeError(f"self.{key} is an Element, only float/int/bool type value can be assigned to it, "
                                f"instead of {type(value)}.")
        if isinstance(value, Element) and key not in self.elements:
            raise RuntimeError("Please use register_element() function to add Element.")
        attrs = self.__dict__
        if key in attrs and isinstance(value, Element):
            raise RuntimeError(f'`{key}` has been registered as an attribute, cannot be registered as an Element!')
        object.__setattr__(self, key, value)

    # 当调用 __getattribute__ 没有找到时才会触发这个, 保留这个的目的只是为了防止 ide 的 warning
    def __getattr__(self, name: str) -> Element:
        if 'elements' in self.__dict__:
            elements = self.__dict__['elements']
            if name in elements:
                return elements[name]
        raise AttributeError("`{}` object has no attribute `{}`.".format(type(self).__name__, name))

    def _wrap_update(self, update):
        @functools.wraps(update)
        def _wrap_update(*args, **kwargs):
            self.check_backend(*args, **kwargs)
            self._cannot_change_element = True
            self._updated = True
            return update(*args, **kwargs)

        return _wrap_update

    def check_backend(self, *args, **kwargs):
        """
        根据传入的参数的类型选择当前需要的 backend
        """
        if not self.backend.is_specified():
            _args = []
            for arg in args:
                _args.append(arg)
            for arg in kwargs.values():
                _args.append(arg)
            self.backend.choose_real_backend(_args)

    @contextmanager
    def sync(self, recover=True, aggregate=False):
        """
        在这个上下文下， :meth:`Metric` 会自动先同步需要同步操作的 element 。当 ``recover`` 为 ``True`` 时，在退出环境的时候，会重新将 element 的
        值恢复到计算前的值。
        """
        keep_value = {}
        if aggregate:
            for name, element in self.elements.items():
                # 保存过去的值
                keep_value[name] = element.get_scalar()
                # 聚合结果
                element.aggregate()

        yield

        if recover and aggregate:
            for name, element in self.elements.items():
                # 恢复结果
                if name in keep_value:
                    element.fill_value(value=keep_value.get(name))

    @abstractmethod
    def update(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def get_metric(self) -> dict:
        raise NotImplementedError()

    def set_auto_aggregate_when_get_metric(self, flag: bool):
        """
        设置是否在 :meth:`get_metric` 的时候自动 aggregate

        """
        self.aggregate_when_get_metric = flag

    def tensor2numpy(self, tensor) -> np.array:
        """
        将 ``tensor`` 向量转为 :class:`numpy.array` 类型变量。

        :param tensor:
        :return:
        """
        return self.backend.tensor2numpy(tensor)

    def to(self, device):
        """
        将所有的 element 变量移动到 ``device`` 设备上

        :param device:
        :return:
        """
        for element in self.elements.values():
            element.to(device)

    def all_gather_object(self, obj, group=None)->List:
        """
        给定 ``obj`` 将各个 rank 上的 ``obj`` 汇总到每个 ``obj`` 上。返回一个 list 对象，里面依次为各个 rank 对应的 ``obj`` 。

        :param obj: 需要汇总的对象，必须是个 pickable 的对象。
        :param group:
        :return: -> List[obj0, obj1, ...] 其中 obj0 是rank 0 上的 obj；obj1 是 rank 1 上的 obj...
        """
        if self.aggregate_when_get_metric:
            return self.backend.all_gather_object(obj, group=group)
        return [obj]
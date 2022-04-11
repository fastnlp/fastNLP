import inspect
from typing import List, Optional, Dict, Sequence
from collections import defaultdict

__all__ = [
    'CallbackManager'
]

from .callback_events import Events
from .callback import Callback
from .checkpoint_callback import TrainerCheckpointCallback
from .progress_callback import ProgressCallback, choose_progress_callback
from fastNLP.core.log import logger


def _transfer(func):
    r"""
    装饰器，将对CallbackManager的调用转发到各个Callback子类.
    需要注意这里的 wrapper 内的函数不会运行 `func` 本身，因此如果有什么需要直接在 callback 函数内运行的代码，请放在 TrainerCallback 内；
    """

    def wrapper(manager, *arg, **kwargs):
        manager.callback_counter[func.__name__] += 1  # 给实际被调用的 callback_fn 的计数加 1；
        returns = []
        for callback_fn in manager.callback_fns[func.__name__]:
            returns.append(callback_fn(*arg, **kwargs))
        return returns
    return wrapper


class CallbackManager:
    r"""
    用来管理训练过程中的所有的 callback 实例；
    """
    all_callbacks: List[Callback]
    class_callbacks: Optional[List[Callback]]  # 用来保留原始的类callback；
    callback_fns: dict

    def __init__(self, callbacks: Optional[List[Callback]], progress_bar='auto'):
        r"""
        注意 callback 的调用顺序：
            1. 通过函数修饰器 `Trainer.on` 添加的 callback 函数；
            2. 通过 `Trainer` 的参数 `callbacks` 添加的 callback 类；
            3. 通过 `Trainer.add_callback_fn` 添加的 callback 函数；

        :param callbacks: 初始化时可以传入的一系列 callback 类，通常为用户在初始化 'Trainer' 时直接传入的 callback 类；
        """
        self._has_trainer_checkpoint = False

        _has_progress_callback = False
        _callbacks = []
        if callbacks is not None:
            if isinstance(callbacks, Callback):
                callbacks = [callbacks]
            if not isinstance(callbacks, Sequence):
                raise ValueError("Parameter `callbacks` should be type 'List' or 'Tuple'.")
            callbacks = list(callbacks)
            for _callback in callbacks:
                if not isinstance(_callback, Callback):
                    raise TypeError(f"callbacks must be of Callback type, instead of `{type(_callback)}`")
                if isinstance(_callback, ProgressCallback):
                    _has_progress_callback = True
            _callbacks += callbacks
        if not _has_progress_callback:
            # 添加 progress callback
            progress_callback = choose_progress_callback(progress_bar=progress_bar)
            if progress_callback is None:
                logger.info("There is no progress bar, Trainer will not output training progress.")
            else:
                _callbacks.append(progress_callback)
        self.callback_fns = defaultdict(list)
        # 因为理论上用户最多只能通过 'trainer.on_train_begin' 或者 'trainer.callback_manager.on_train_begin' 来调用，即其是没办法
        #  直接调用具体的某一个 callback 函数，而不调用其余的同名的 callback 函数的，因此我们只需要记录具体 Event 的时机即可；
        self.callback_counter = defaultdict(lambda: 0)
        if len(_callbacks):
            # 这一对象是为了保存原始的类 callback 对象来帮助用户进行 debug，理论上在正常的使用中你并不会需要它；
            self.class_callbacks = _callbacks
        else:
            self.class_callbacks: Optional[List[Callback]] = []

        # 预跑需要拿到每一个被 `Filter` 修饰的函数的 `Filter` 实例，从而在预跑结束后重置它们的内部状态；
        self._callback_filters = []  # [(callback_name, fn_name, filter 实例), ]

        # 保留所有 callback 的引用，用于断点重训；包括全部的三种callback：函数修饰器 callback；类 callback；纯函数 callback；
        # 因为所有的 callback 都是通过函数 `self.add_one_callback` 添加，因此我们选择在其下进行添加；
        # 一个比较重要的概念在于在训练过程运行的时候，两个 callback 的 callback_name 可以是一样的，并且理论上不会造成任何影响；但是当
        #  `on_load_checkpoint` 时，我们需要处理两个 callback_name 一样这种情况了；
        # 因此这里的 `all_callbacks` 为了避免正常训练过程的运行，只能是一个 List，而不能是一个 dict，`_callback_filters` 也是一样；
        self.all_callbacks = []

    def initialize_class_callbacks(self):
        r"""
        在实际的运行过程中，我们是将具体的一个 callback 实例拆分为单独的一个个 callback 函数，然后将它们加在一个字典里，该字典的键值就是
         一个个 callback 时机，也就是 `Events` 的类别；
        如果一个 callback 类的 callback 函数并不具备任何作用，我们实际并不会将其加在字典当中；

        :param callbacks:
        :return:
        """
        for each_callback in self.class_callbacks:
            if isinstance(each_callback, TrainerCheckpointCallback):
                self._has_trainer_checkpoint = True
            self.dissect_one_callback(each_callback)

    def dissect_one_callback(self, callback: Callback):
        r"""
        将具体的一个 callback 实例的所有 callback 函数拆分后按时机插入到字典中；

        :param callback: 一个具体的 callback 实例；
        """
        self.all_callbacks.append(callback)
        for name, member in Events.__members__.items():
            _fn = getattr(callback, member.value)
            if inspect.getsource(_fn) != inspect.getsource(getattr(Callback, member.value)):
                self.callback_fns[member.value].append(_fn)
                self.extract_callback_filter_state(callback.callback_name, _fn)

    def extract_callback_filter_state(self, callback_name, callback_fn):
        r"""
        将一个具体的 callback 函数的 filter 的状态抽取出来；
        """
        if hasattr(callback_fn, "__fastNLP_filter__"):
            # 注意我们的 `Filter` 使用了 `@wraps` 来保证被修饰的函数的 `__name__` 属性仍旧是其真实的名字；
            self._callback_filters.append((callback_name, callback_fn.__name__, callback_fn.__fastNLP_filter__))

    def on_save_checkpoint(self, trainer) -> Dict:
        r"""
        用于断点重训的 callback 的保存函数；
        该函数主要涉及两个方面：
            1. callback 的状态的保存；我们会调用每一个 callback 的 `on_save_checkpoint` 方法，该方法应当返回一个字典，其中包含着
             断点重训应当保存的状态；
            2. 每一个具体的 callback 函数的 filter 的状态；

        :return: 一个包含上述内容的字典；
            {
                "callback_name_1": {
                    "states": {...},
                    "filter_states": {"on_train_begin": filter1.state_dict(), ...}
                }
            }
        """

        states = {}
        # 1. 每一个 callback 的状态；
        # 如果有两个 callback 的 name 相同，那么我们只会保存第一个；
        _duplicated_callbacks = []
        for each_callback in self.all_callbacks:
            if each_callback.callback_name in states:
                _duplicated_callbacks.append(each_callback.callback_name)
                # 对于 callback_name 有重复的 callback，我们仍旧会调用其 `on_save_checkpoint` 函数，就如同调用其它 callback 函数
                #  一样，但是其结果并不会存储在 states 中返回；
                each_callback.on_save_checkpoint(trainer)
            else:
                states[each_callback.callback_name] = {}
                states[each_callback.callback_name]["states"] = each_callback.on_save_checkpoint(trainer)

        if len(_duplicated_callbacks) > 0:
            logger.warning(f"Notice these callbacks' `callback_name` are duplicated: {_duplicated_callbacks}, "
                           f"and we will only save the first callback's state we meet.")

        # 2. 每一个具体的 callback 函数的 filter 的状态；
        _record_duplicated_callback_names = set()
        for each_callback_filters in self._callback_filters:
            if each_callback_filters[0] not in _record_duplicated_callback_names:
                _record_duplicated_callback_names.add(each_callback_filters[0])
                states[each_callback_filters[0]]["filter_states"][each_callback_filters[1]] = each_callback_filters[2].state_dict()

        # 3. 保存 callback_counter；
        # callback_counter 不应当被保存，因为其在断点重训时会由新的 callback_manager 重新初始化；
        # 对于断点重训，我们不会保存 Trainer 的所有参数，例如 batch_step_fn；如果在断点重训时重新初始化 Trainer 发现 batch_step_fn
        # 不为 None，那么 Trainer 就会调用实际的 check_batch_step_fn 函数，从而需要 callback_counter 为全新的状态；

        return states

    def on_load_checkpoint(self, trainer, states: Dict):
        r"""
        用于断点重训的加载函数；
        对应于断点重训的保存函数；

        :param trainer: `Trainer`
        :param states: 见 `on_save_checkpoint` 函数的返回值；
        """

        # 1. 先恢复每一个具体的 callback 函数的 filter 的状态；
        # self._callback_filters 是当前的 Trainer 的 callback 的 filter 状态，是我们要去维护的对象；
        _already_loaded_callback_names = set()
        _duplicated_callback_names = set()
        for each_callback_filters in self._callback_filters:
            if each_callback_filters[0] in states:
                if each_callback_filters[0] not in _already_loaded_callback_names:
                    _already_loaded_callback_names.add(each_callback_filters[0])
                    each_callback_filters[2].load_state_dict(states[each_callback_filters[0]]["filter_states"][each_callback_filters[1]])
                else:
                    _duplicated_callback_names.add(each_callback_filters[0])

        if len(_duplicated_callback_names) > 0:
            logger.warning(f"Notice these callbacks' `callback_name` are duplicated: {_duplicated_callback_names}, "
                           f"and we will only load the first callback's state we meet.")

        # 2. 再恢复每一个 callback 的单独的状态；
        # 每一个我们自己提供的类 callback，都需要重写其特定的 `callback_name` 方法，保证如果两个 callback 的 callback_name 一样，
        #  那么它们就应该是同一个对象；
        _already_loaded_callback_names = set()
        for each_callback in self.all_callbacks:
            if each_callback.callback_name in states and each_callback.callback_name not in _already_loaded_callback_names:
                _already_loaded_callback_names.add(each_callback.callback_name)
                # 这里要注意，我们已经确保每一个 callback 的 `on_load_checkpoint` 函数拿到的就是其自己的状态；
                each_callback.on_load_checkpoint(trainer, states[each_callback.callback_name]["states"])
            else:
                each_callback.on_load_checkpoint(trainer, None)

    @property
    def has_trainer_checkpoint(self) -> bool:
        return self._has_trainer_checkpoint

    @_transfer
    def on_after_trainer_initialized(self, trainer):
        pass

    @_transfer
    def on_sanity_check_begin(self, trainer):
        pass

    @_transfer
    def on_sanity_check_end(self, trainer):
        pass

    @_transfer
    def on_train_begin(self, trainer):
        pass

    @_transfer
    def on_train_end(self, trainer):
        pass

    @_transfer
    def on_train_epoch_begin(self, trainer):
        pass

    @_transfer
    def on_train_epoch_end(self, trainer):
        pass

    @_transfer
    def on_fetch_data_begin(self, trainer):
        pass

    @_transfer
    def on_fetch_data_end(self, trainer):
        pass

    @_transfer
    def on_train_batch_begin(self, trainer, batch, indices=None):
        pass

    @_transfer
    def on_train_batch_end(self, trainer):
        pass

    @_transfer
    def on_exception(self, trainer, exception):
        pass

    @_transfer
    def on_save_model(self, trainer):
        pass

    @_transfer
    def on_load_model(self, trainer):
        pass

    @_transfer
    def on_before_backward(self, trainer, outputs):
        pass

    @_transfer
    def on_after_backward(self, trainer):
        pass

    @_transfer
    def on_before_optimizer_step(self, trainer, optimizers):
        pass

    @_transfer
    def on_before_zero_grad(self, trainer, optimizers):
        pass

    @_transfer
    def on_validate_begin(self, trainer):
        pass

    @_transfer
    def on_validate_end(self, trainer, results):
        pass

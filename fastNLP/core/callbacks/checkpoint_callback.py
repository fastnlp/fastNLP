import os
from typing import Union, Optional, Callable, Dict, Sequence
from pathlib import Path
from functools import partial
from time import sleep

__all__ = [
    'CheckpointCallback'
]

import fastNLP
from .callback import Callback, Filter
from fastNLP.core.callbacks.utils import _get_monitor_value
from fastNLP.core.log import logger
from fastNLP.envs import FASTNLP_LAUNCH_TIME
from fastNLP.core.utils import synchronize_safe_rm, synchronize_mkdir


class CheckpointCallback(Callback):
    """
    1. 因为只有 'Trainer' 才有 callback，因此评测 metric 实际上就是 validate 时干的事情；
    2. 默认 'save_last' 为 True，即 model_checkpoint 的默认逻辑是在每一个 epoch 下保存最后的一个模型，模型名字为 last.pth.tar；
    3. 理论上一个 model_checkpoint 的实例只会负责一个 monitor 的监视，如果用户在训练过程中指定了多个 monitor 的监视，例如 "acc1",
    "acc2", ... 那么我们会为用户创建多个 model_checkpoint 的实例；
    4. 理论上，在实际保存的过程中，topk 模式和 固定频率保存的模式是完全独立的，我们确实应当采取一些措施至少保证两者的名字不一样；
    """

    def __init__(
            self,
            monitor,
            is_trainer_checkpoint: Optional[bool] = False,

            save_folder: Optional[Union[str, Path]] = None,

            save_every_n_epochs: Optional[int] = None,
            save_every_n_global_batches: Optional[int] = None,
            save_last: bool = True,
            save_topk: Optional[int] = None,
            save_on_exception: Optional[Union[BaseException, Sequence[BaseException]]] = None,

            larger_better: bool = True,
            only_state_dict: bool = True,

            model_save_fn: Optional[Callable] = None,

            **kwargs,
    ):
        if monitor is None and save_topk is not None:
            raise ValueError("Parameter `monitor` must be set when you want to use 'save_topk'.")

        if monitor is not None and not isinstance(monitor, str):
            raise ValueError("Parameter `monitor` should be of 'str' type.")

        if not isinstance(is_trainer_checkpoint, bool):
            raise TypeError("Parameter 'is_trainer_checkpoint' can only be `bool` type.")

        if save_folder is None:
            logger.warning(
                "Parameter `path` is None, and we will use the current work directory to find and load your model.")
            save_folder = Path.cwd()
        if not save_folder.exists():
            raise NotADirectoryError(f"Path '{save_folder.absolute()}' is not existed!")
        elif save_folder.is_file():
            raise ValueError("Parameter `save_folder` should be a directory instead of a file.")

        if save_every_n_epochs is not None:
            if not isinstance(save_every_n_epochs, int) or save_every_n_epochs < 1:
                raise ValueError("parameter save_after_epoch_num should be an int and greater than or equal to 1.")

            # 突然发现有一个骚操作在于 'Filter' 内部记载的状态值例如 'num_called' 是这个类全局的，而每次调用 __call__ 中输入的
            # 函数却是及时传入的，也就是说，我们可以保证 'Filter' 的正常控制频率的逻辑，然后每一次运行的函数都不一样；
            self._filter_every_n_epochs = Filter(every=save_every_n_epochs)

        if save_every_n_global_batches is not None:
            if not isinstance(save_every_n_global_batches, int) or save_every_n_global_batches < 1:
                raise ValueError(
                    "parameter save_every_n_global_batches should be an int and greater than or equal to 1.")
            self._filter_every_n_global_batches = Filter(every=save_every_n_global_batches)

        if save_topk is not None:
            if not isinstance(save_topk, int) or save_topk < 1:
                raise ValueError("parameter save_topk should be an int and greater than or equal to 1.")

        if save_on_exception is not None:
            if not isinstance(save_on_exception, Sequence):
                save_on_exception = [save_on_exception]

            for exception in save_on_exception:
                if not issubclass(exception, BaseException):
                    raise TypeError("Each exception in parameter `save_on_exception` can only be "
                                    "`BaseException` type.")

        self.monitor = monitor
        self.is_trainer_checkpoint = is_trainer_checkpoint
        self.save_folder = Path(save_folder)
        self.save_every_n_epochs = save_every_n_epochs
        self.save_every_n_global_batches = save_every_n_global_batches
        self.save_last = save_last
        self.save_topk = save_topk
        self.larger_better = larger_better
        self.only_state_dict = only_state_dict
        self.model_save_fn = model_save_fn
        self.save_on_exception = save_on_exception
        self.kwargs = kwargs

        # 这些参数是专门留给 topk 模式专门使用的；
        self._topk_model = {}
        self._topn = 0  # 表示目前已经保存了几个最好的模型；

        # 因为我们在 `_get_validate_metric` 函数中，当在返回的 `validate_res` 字典中找不到 `monitor` 时，是使用模糊匹配找到的第一个
        #  key 对应的 value 当做结果；但是这样存在的一个问题在于如果用户传入的 metric 返回的 sub_metric 的名字可能会混淆，并且其在下一次
        #  训练的代码中修改了这些 sub_metric 返回的顺序，那么就会导致模糊匹配拿到的 key 和 value 与之前的不是同一个，这显然不是合理的行为；
        # 因此我们通过该变量来表示我们通过模糊匹配拿到的 key；
        self._real_monitor = self.monitor

        # 注意这里应当保证只有进程 0 在执行这个操作，因为当用户使用 python -m torch.distributed.launch 来拉起进程的时候，
        #  FASTNLP_LAUNCH_TIME 在每一个进程上的值是不一样的；
        self.log_filepath = self.save_folder.joinpath(os.environ[FASTNLP_LAUNCH_TIME])
        # 我们只需要保证这个创建文件夹的操作只在进程 0 上进行即可；因为后续的实际的保存操作，其它进程实际并不会去执行；
        synchronize_mkdir(self.log_filepath)

    def on_validate_end(self, trainer, validate_res):
        self._save_topk(trainer, validate_res)

    def on_train_epoch_end(self, trainer: "fastNLP.Trainer"):
        self._save_every_n_epochs(trainer)
        self._save_last(trainer)

    def on_train_batch_end(self, trainer):
        self._save_every_n_global_batches(trainer)

    def on_exception(self, trainer, exception: BaseException):
        if self.save_on_exception is not None and exception.__class__ in self.save_on_exception:
            folder = self._get_checkpoint_real_save_folder(trainer=trainer, topk=False, metric=None)
            folder = folder + f"_{exception.__class__.__name__}"
            self._save_fn(trainer=trainer, topk=False, metric=None, substitute_folder=folder)

    def on_sanity_check_end(self, trainer, sanity_check_res):
        self._get_validate_metric(sanity_check_res)

    def on_save_checkpoint(self, trainer) -> Dict:
        """
        我们需要保存 CheckpointCallback 内部的几个 filter 的状态；
        """
        states = {}
        if self.save_every_n_epochs is not None:
            states["_filter_every_n_epochs"] = self._filter_every_n_epochs.state_dict()
        if self.save_every_n_global_batches is not None:
            states["_filter_every_n_global_batches"] = self._filter_every_n_global_batches.state_dict()
        states["real_monitor"] = self._real_monitor
        return states

    def on_load_checkpoint(self, trainer, states: Optional[Dict]):
        if self.save_every_n_epochs is not None:
            self._filter_every_n_epochs.load_state_dict(states["_filter_every_n_epochs"])
        if self.save_every_n_global_batches is not None:
            self._filter_every_n_global_batches.load_state_dict(states["_filter_every_n_global_batches"])
        self._real_monitor = states["real_monitor"]

    def _save_every_n_epochs(self, trainer: "fastNLP.Trainer"):
        if self.save_every_n_epochs is not None:
            if self.is_trainer_checkpoint:
                _fn_every_n_epochs = trainer.save
            else:
                _fn_every_n_epochs = trainer.save_model
            _fn_every_n_epochs = partial(self._save_fn, trainer, False, None, _fn_every_n_epochs, None)
            _fn_every_n_epochs = self._filter_every_n_epochs(_fn_every_n_epochs)
            _fn_every_n_epochs()

    def _save_every_n_global_batches(self, trainer: "fastNLP.Trainer"):
        if self.save_every_n_global_batches is not None:
            if self.is_trainer_checkpoint:
                _fn_every_n_global_batches = trainer.save
            else:
                _fn_every_n_global_batches = trainer.save_model
            _fn_every_n_global_batches = partial(self._save_fn, trainer, False, None, _fn_every_n_global_batches, None)
            _fn_every_n_global_batches = self._filter_every_n_global_batches(_fn_every_n_global_batches)
            _fn_every_n_global_batches()

    def _save_topk(self, trainer: "fastNLP.Trainer", validate_res: Dict):
        if self.save_topk is not None:
            _metric_value = self._get_validate_metric(validate_res)
            _saved_name = self._get_checkpoint_real_save_folder(trainer=trainer, topk=True, metric=_metric_value)

            _should_save = False
            if self._topn < self.save_topk:
                self._topk_model[_saved_name] = _metric_value
                self._topn += 1
                _should_save = True
            else:
                _least_valuable_model = (min if self.larger_better else max)(self._topk_model,
                                                                             key=lambda x: self._topk_model[x])
                if (self.larger_better and _metric_value > self._topk_model[_least_valuable_model]) or \
                        (self.larger_better is False and _metric_value < self._topk_model[_least_valuable_model]):
                    self._topk_model[_saved_name] = _metric_value
                    _should_save = True
                    self._topk_model.pop(_least_valuable_model)
                    synchronize_safe_rm(self.log_filepath.joinpath(_least_valuable_model))

                assert len(self._topk_model) == self.save_topk == self._topn

            if _should_save:
                self._save_fn(trainer=trainer, topk=True, metric=_metric_value, substitute_folder=_saved_name)

    def _save_last(self, trainer: "fastNLP.Trainer"):
        if self.save_last:
            self._save_fn(trainer=trainer, topk=False, metric=None, substitute_folder="last")

    def _save_fn(self, trainer, topk: bool = False, metric: Optional[Union[int, float]] = None,
                 substitute_fn: Optional[Callable] = None, substitute_folder: Optional[str] = None):
        # 首先根据当前的 epoch 和 batch 在 parent_path/FASTNLP_LAUNCH_TIME 下创建子文件夹 epoch-batch-monitor 或者
        #  epoch-batch-monitor-monitor_value；
        if substitute_folder is None:
            folder = self.log_filepath.joinpath(self._get_checkpoint_real_save_folder(trainer, topk, metric))
        else:
            folder = self.log_filepath.joinpath(substitute_folder)

        synchronize_mkdir(folder)

        # 然后再调用 trainer 的 save_model（用于保存模型）或者 save（用于断点重训）函数；
        if substitute_fn is not None:
            _fn = substitute_fn
        else:
            if self.is_trainer_checkpoint:
                _fn = trainer.save
            else:
                _fn = trainer.save_model
        _fn(
            folder=folder,
            only_state_dict=self.only_state_dict,
            model_save_fn=self.model_save_fn,
            **self.kwargs
        )

    def _get_validate_metric(self, res: Dict):
        """
        该函数用于从 `Evaluator` 的结果中找到属于当前 CheckpointCallback 的 metric result（根据 monitor）；
        如果用户输入在 res 中没有找到，我们会查询所有的 validate 结果字典的键值，根据 最长公共字符串 匹配，使用最长匹配的结果值；
        :param res:
        :return:
        """
        use_monitor, value = _get_monitor_value(monitor=self.monitor, real_monitor=self._real_monitor, res=res)
        self._real_monitor = use_monitor
        return value

    def _get_checkpoint_real_save_folder(self, trainer: "fastNLP.Trainer", topk: bool = False,
                                         metric: Optional[Union[int, float]] = None) -> str:
        """
        获取当前保存模型的真正地名字；
        metric 参数仅当 mode 为 'topk' 时起作用；
        """
        cur_epoch_idx = trainer.cur_epoch_idx
        global_forward_batches = trainer.global_forward_batches
        _other = ""
        if topk:
            _other = f"_{metric}"
        return f"epoch_{cur_epoch_idx}-global_batch_{global_forward_batches}-{self._real_monitor}{_other}"

    @property
    def callback_name(self):
        """
        通过该值决定两个 CheckpointCallback 实例是否可以共用断点重训的状态；
        :return:
        """
        return f"monitor-{self.monitor}#trainer_checkpoint-{self.is_trainer_checkpoint}#only_state_dict-{self.only_state_dict}"



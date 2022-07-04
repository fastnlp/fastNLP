__all__ = [
    'CheckpointCallback'
]

from typing import Union, Optional, Callable, Dict, Sequence
from pathlib import Path
import sys

from fastNLP.core.log import logger
from .topk_saver import TopkSaver
from .callback import Callback
from ..utils.exceptions import EarlyStopException


class CheckpointCallback(Callback):
    """
    保存 checkpoint 的  callback ，其保存的文件目录以及文件名命名规则如下::

        - folder/
            - YYYY-mm-dd-HH_MM_SS_fffff/  # 自动根据当前脚本的启动时间创建的
                - {save_object}-epoch_{epoch_idx}/  # 满足 every_n_epochs 条件保存的模型
                - {save_object}-epoch_{epoch_idx}-batch_{global_batch_idx}/  # 满足 every_n_batches 保存的模型
                - {save_object}-last/  # 最后一个 epoch 的保存
                - {save_object}-epoch_{epoch_idx}-batch_{global_batch_idx}-exception_{exception_type}/  # exception时保存。
                - {save_object}-epoch_{epoch_idx}-batch_{global_batch_idx}-{monitor}_{monitor_value}/  # 满足topk条件存储文件名

    ``model_save_fn`` 为 ``None`` ，则以上每个 folder 中，将生成 fastnlp_model.pkl.tar 文件。若 ``model_save_fn`` 不为 ``None``，
    则 fastNLP 将 folder 绝对路径传递给该函数，fastNLP 在该 folder 下不进行模型保存。默认情况下，本 checkpoint 只保存了 model
    的状态；如还需保存 Trainer 的状态以断点重训的话，请使用 ``save_object='trainer'`` 。

    :param monitor: 监控的 metric 值。

        * 为 ``None``
          将尝试使用 :class:`~fastNLP.core.controllers.Trainer` 中设置 `monitor` 值（如果有设置）。
        * 为 ``str``
          尝试直接使用该名称从 ``evaluation`` 结果中寻找，如果在 ``evaluation`` 结果中没有找到完全一致的名称，将
          使用 最长公共字符串算法 从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor`` 。
        * 为 :class:`Callable`
          接受参数为 ``evaluation`` 的结果(字典类型)，返回一个 ``float`` 值作为 ``monitor`` 的结果，如果当前结果中没有相关
          的 ``monitor`` 值请返回 ``None`` 。

    :param folder: 保存的文件夹，fastNLP 将在该文件下以时间戳创建子文件夹，并在里面保存。因此不同次运行可以将被保存到不同的
        时间戳文件夹中。如果为 None ，默认使用当前文件夹。
    :param every_n_epochs: 多少个 epoch 保存一次。
    :param every_n_batches: 多少个 batch 保存一次。
    :param last: 如果为 ``True`` ，将在每次 epoch 运行结束都保存一次，会覆盖之前的保存。如果为 ``False`` 则不会保存 ``{save_object}-last`` 文件
    :param topk: 保存 monitor 结果中的 ``topk`` 个。
    :param on_exceptions: 在出异常信息时，是否保存。传入需要捕获的异常的类。默认将捕获 :class:`~fastNLP.core.callbacks.EarlyStopException` 。
    :param larger_better: monitor 的值是否时越大越好。
    :param only_state_dict: 保存模型时是否只保存 state_dict 。当 ``model_save_fn`` 不为 ``None`` 时，该参数无效。
    :param model_save_fn: 个性化的保存函数，当触发保存操作时，就调用这个函数，这个函数应当接受一个文件夹作为参数，不返回任何东西。
        如果传入了 ``model_save_fn`` 函数，fastNLP 将不再进行模型相关的保存。在多卡场景下，我们只在 rank 0 上会运行该函数。
    :param save_object: 可选 ``['trainer', 'model']`` ，表示在保存时的保存对象为 ``trainer+model`` 还是 只是 ``model`` 。如果
        保存 ``trainer`` 对象的话，将会保存 :class:`~fastNLP.core.controllers.Trainer` 的相关状态，可以通过 :meth:`Trainer.load_checkpoint` 加载该断
        点继续训练。如果保存的是 ``Model`` 对象，则可以通过 :meth:`Trainer.load_model` 加载该模型权重。
    :param save_evaluate_results: 是否保存 evaluate 的结果。如果为 ``True`` ，在保存 topk 模型的 folder 中还将额外保存一个
        ``fastnlp_evaluate_results.json`` 文件，记录当前的 results。仅在设置了 ``topk`` 的场景下有用，默认为 ``True`` 。
    :param kwargs:
    """
    def __init__(self, folder: Optional[Union[str, Path]] = None, every_n_epochs: Optional[int] = None,
                 every_n_batches: Optional[int] = None, last: bool = False, topk: int = 0,
                 on_exceptions: Optional[Union[BaseException, Sequence[BaseException]]] = (EarlyStopException),
                 monitor: Optional[Union[str, Callable]] = None, larger_better: bool = True,
                 only_state_dict: bool = True, model_save_fn: Optional[Callable] = None, save_object: str = 'model',
                 save_evaluate_results=True, **kwargs):
        super().__init__()
        if every_n_epochs is not None:
            if not isinstance(every_n_epochs, int) or every_n_epochs < 1:
                raise ValueError("Parameter `every_n_epochs` should be an int and greater than or equal to 1.")
        else:
            every_n_epochs = sys.maxsize  # 使得没有数字可以整除

        if every_n_batches is not None:
            if not isinstance(every_n_batches, int) or every_n_batches < 1:
                raise ValueError("Parameter `every_n_batches` should be an int and greater than or equal to 1.")
        else:
            every_n_batches = sys.maxsize  # 使得没有数字可以整除

        if on_exceptions is not None:
            if not isinstance(on_exceptions, Sequence):
                on_exceptions = [on_exceptions]

            for exception in on_exceptions:
                if not issubclass(exception, BaseException):
                    raise TypeError("Each exception in parameter `on_exception` can only be "
                                    "`BaseException` type.")
        else:
            on_exceptions = []

        self.topk_saver = TopkSaver(topk=topk, monitor=monitor, larger_better=larger_better, folder=folder,
                                    save_object=save_object, only_state_dict=only_state_dict, model_save_fn=model_save_fn,
                                    save_evaluate_results=save_evaluate_results, **kwargs)
        self.topk_saver.log_name = self.__class__.__name__

        self.topk = topk
        self.save_object = save_object

        self.every_n_epochs = every_n_epochs
        self.every_n_batches = every_n_batches
        self.last = last
        self.exceptions = on_exceptions

    @property
    def need_reproducible_sampler(self) -> bool:
        return self.save_object == 'trainer'

    def on_after_trainer_initialized(self, trainer, driver):
        if self.topk_saver.topk_queue:  # 需要设置 monitor
            if self.topk_saver.monitor is None:
                self.topk_saver.set_monitor(monitor=trainer.monitor, larger_better=trainer.larger_better)
        if self.topk_saver.topk_queue and trainer.evaluator is None:
            logger.warning(f"You set `topk={self.topk}`, but `evaluate_dataloaders` is not set in Trainer.")

    def on_evaluate_end(self, trainer, results):
        # 如果发生了保存，则返回的 folder 不为 None
        folder = self.topk_saver.save_topk(trainer, results)

    def on_train_epoch_end(self, trainer: "fastNLP.Trainer"):
        if trainer.cur_epoch_idx % self.every_n_epochs == 0:
            folder_name = f'{self.save_object}-epoch_{trainer.cur_epoch_idx}'
            self.topk_saver.save(trainer, folder_name=folder_name)
        if self.last:
            folder_name = f'{self.save_object}-last'
            self.topk_saver.save(trainer, folder_name=folder_name)

    def on_train_batch_end(self, trainer):
        if trainer.global_forward_batches % self.every_n_batches == 0:
            folder_name = f'{self.save_object}-epoch_{trainer.cur_epoch_idx}-batch_{trainer.global_forward_batches}'
            self.topk_saver.save(trainer, folder_name=folder_name)

    def on_exception(self, trainer, exception: BaseException):
        if exception.__class__ in self.exceptions:
            folder_name = f'{self.save_object}-epoch_{trainer.cur_epoch_idx}-batch_{trainer.global_forward_batches}-' \
                          f'exception_{exception.__class__.__name__}'
            self.topk_saver.save(trainer, folder_name=folder_name)

    def on_save_checkpoint(self, trainer) -> Dict:
        states = {}
        states['topk_saver'] = self.topk_saver.state_dict()
        return states

    def on_load_checkpoint(self, trainer, states: Optional[Dict]):
        topk_saver_states = states['topk_saver']
        self.topk_saver.load_state_dict(topk_saver_states)


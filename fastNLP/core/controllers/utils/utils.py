from typing import Dict

from fastNLP.core.callbacks import CallbackManager
from .state import TrainerState
from fastNLP.core.utils.utils import _check_valid_parameters_number

__all__ = []

class TrainerEventTrigger:
    r"""
    为了避免在训练流程中调用 callback 函数中写成类似 `'trainer.callback_manager.on_train_begin'` 的形式，我们选择单独为 ``Trainer``
    抽象一层，然后一些特殊的操作可以在这里进行，例如我们通过 :meth:`on_validate_end` 来通知所有的 ``CheckpointCallback`` 实例在当前的 step 后保存
    模型。
    """
    callback_manager: CallbackManager
    trainer_state: TrainerState

    def on_after_trainer_initialized(self, driver):
        self.callback_manager.on_after_trainer_initialized(self, driver)

    def on_sanity_check_begin(self):
        self.callback_manager.on_sanity_check_begin(self)

    def on_sanity_check_end(self, sanity_check_res):
        self.callback_manager.on_sanity_check_end(self, sanity_check_res)

    def on_train_begin(self):
        self.callback_manager.on_train_begin(self)

    def on_train_end(self):
        self.callback_manager.on_train_end(self)

    def on_train_epoch_begin(self):
        self.callback_manager.on_train_epoch_begin(self)

    def on_train_epoch_end(self):
        self.callback_manager.on_train_epoch_end(self)

    def on_fetch_data_begin(self):
        self.callback_manager.on_fetch_data_begin(self)

    def on_fetch_data_end(self):
        self.callback_manager.on_fetch_data_end(self)

    def on_train_batch_begin(self, batch, indices=None):
        self.callback_manager.on_train_batch_begin(self, batch, indices)

    def on_train_batch_end(self):
        self.callback_manager.on_train_batch_end(self)

    def on_exception(self, exception):
        self.callback_manager.on_exception(self, exception)

    def on_save_model(self):
        self.callback_manager.on_save_model(self)

    def on_load_model(self):
        self.callback_manager.on_load_model(self)

    def on_save_checkpoint(self) -> Dict:
        return self.callback_manager.on_save_checkpoint(self)

    def on_load_checkpoint(self, states):
        self.callback_manager.on_load_checkpoint(self, states)

    def on_before_backward(self, outputs):
        self.callback_manager.on_before_backward(self, outputs)

    def on_after_backward(self):
        self.callback_manager.on_after_backward(self)

    def on_before_optimizers_step(self, optimizers):
        self.callback_manager.on_before_optimizers_step(self, optimizers)

    def on_after_optimizers_step(self, optimizers):
        self.callback_manager.on_after_optimizers_step(self, optimizers)

    def on_before_zero_grad(self, optimizers):
        self.callback_manager.on_before_zero_grad(self, optimizers)

    def on_after_zero_grad(self, optimizers):
        self.callback_manager.on_after_zero_grad(self, optimizers)

    def on_evaluate_begin(self):
        self.callback_manager.on_evaluate_begin(self)

    def on_evaluate_end(self, results):
        self.trainer_state.save_on_this_step = True
        self.callback_manager.on_evaluate_end(self, results)


class _TruncatedDataLoader:
    r"""
    ``_TruncatedDataLoader`` 用于实现 ``Trainer`` 和 ``Evaluator`` 中的 '预跑' 和 '假跑' 功能：

        1. 预跑 是针对 trainer 的验证而言的，即我们在正式的训练前会先使用 trainer 内置的 evaluator（如果不为 None）评测数量非常少的数据，
        来检验用户的 metric 和 evaluate_dataloader 以及模型是否能够合作完成正确的评测过程；
        2. 假跑 的意思是改变每一个 epoch 中训练或者评测的实际的 batch 的数量，例如改成 10，来让模型快速地迭代整体的训练或者评测流程，来查看
        整体的过程的正确性；

    ``_TruncatedDataLoader`` 的实现也非常简单，我们在该类中内置一个计数器，当迭代器的迭代数量达到指定数值后 ``raise StopIteration``；

    :param dataloader: 可迭代的 dataloader 。
    :param num_batches: 迭代多少个 batch 就停止。
    """
    def __init__(self, dataloader, num_batches: int):

        self.dataloader = dataloader
        self._num_batches = min(num_batches, len(dataloader))
        self._count = 0

    def __len__(self):
        r"""
        为了在外部调用 `len` 方法时正确地返回当前会迭代的长度；
        """
        return self._num_batches

    def __iter__(self):
        # 将初试的 `dataloader` 转换成一个 `Iterator` 的逻辑应该放在这里，即只有当外界真正的调用 iter(dataloader) 的时候才需要返回一个 Iterator；
        # TODO 测试一下
        self._iterator = iter(self.dataloader)
        self._count = 0
        return self

    def __next__(self):
        if self._count >= self._num_batches:
            raise StopIteration
        self._count += 1
        # 注意 dataloader 数据不足时会自己本身触发 `StopIteration`；
        return next(self._iterator)

    def __getattr__(self, item):
        return getattr(self.dataloader, item)


def check_evaluate_every(evaluate_every):
    r"""
    检验用户传入的 ``evaluate_every`` 参数是否合法；

    ``evaluate_every`` 的使用详见 ``Trainer`` 的 ``evaluate_every`` 参数；

    主要在于当参数 ``evaluate_every`` 是一个 Callable 的函数时，需要保证其参数的正确性；
    """
    if not callable(evaluate_every) and (not isinstance(evaluate_every, int) or evaluate_every == 0):
        raise ValueError("Parameter 'evaluate_every' should be set to 'int' type and either < 0 or > 0.")
    if callable(evaluate_every):
        _check_valid_parameters_number(evaluate_every, expected_params=['trainer'])

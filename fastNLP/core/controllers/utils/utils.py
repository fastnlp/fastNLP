import inspect
from typing import Dict

from fastNLP.core.callbacks import CallbackManager
from .state import TrainerState
from fastNLP.core.utils.utils import _check_valid_parameters_number


class TrainerEventTrigger:
    """
    为了避免在训练流程中调用 callback 函数中写成类似 'trainer.callback_manager.on_train_begin' 的形式，我们选择单独抽象为 'Trainer'
     抽象一层，然后一些特殊的操作可以在这里进行，例如我们通过 `on_validate_end` 来通知所有的 'CheckpointCallback' 实例在当前的 step 后保存
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
    def __init__(self, dataloader, num_batches: int):
        """
        限制

        :param dataloader: 可迭代的 dataloader 。
        :param num_batches: 迭代多少个 batch 就停止。
        """
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
    if not callable(evaluate_every) and (not isinstance(evaluate_every, int) or evaluate_every == 0):
        raise ValueError("Parameter 'evaluate_every' should be set to 'int' type and either < 0 or > 0.")
    if callable(evaluate_every):
        _check_valid_parameters_number(evaluate_every, expected_params=['trainer'])

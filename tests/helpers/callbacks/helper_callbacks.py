"""
该模块用于实现一些帮助我们在测试的 callback 类；
"""

from fastNLP.core.callbacks.callback import Callback


class RecordLossCallback(Callback):
    """
    通过该 callback 来测试模型的训练是否基本正常；
    """
    def __init__(self, loss_threshold: float):
        self.loss = None
        self.loss_threshold = loss_threshold
        self.loss_begin_value = None

    def on_before_backward(self, trainer, outputs):
        loss = trainer.extract_loss_from_outputs(outputs)
        loss = trainer.driver.tensor_to_numeric(loss)
        self.loss = loss
        if self.loss_begin_value is None:
            self.loss_begin_value = loss

    def on_train_end(self, trainer):
        assert self.loss < self.loss_begin_value
        if self.loss_threshold is not None:
            assert self.loss < self.loss_threshold


class RecordMetricCallback(Callback):
    """
    通过该 callback 来测试带有 metrics 的 Trainer 是否训练测试正确；
    """
    def __init__(self, monitor: str, metric_threshold: float, larger_better: bool):
        self.monitor = monitor
        self.larger_better = larger_better
        self.metric = None
        self.metric_threshold = metric_threshold
        self.metric_begin_value = None

    def on_validate_end(self, trainer, results):
        self.metric = results[self.monitor]
        if self.metric_begin_value is None:
            self.metric_begin_value = self.metric

    def on_train_end(self, trainer):
        if self.larger_better:
            assert self.metric >= self.metric_begin_value
            assert self.metric > self.metric_threshold
        else:
            assert self.metric <= self.metric_begin_value
            assert self.metric < self.metric_threshold


class RecordTrainerEventTriggerCallback(Callback):
    """
    测试每一个 callback 是否在 trainer 中都得到了调用；
    """
    def on_after_trainer_initialized(self, trainer, driver):
        print("on_after_trainer_initialized")

    def on_sanity_check_begin(self, trainer):
        print("on_sanity_check_begin")

    def on_sanity_check_end(self, trainer, sanity_check_res):
        print("on_sanity_check_end")

    def on_train_begin(self, trainer):
        print("on_train_begin")

    def on_train_end(self, trainer):
        print("on_train_end")

    def on_train_epoch_begin(self, trainer):
        if trainer.cur_epoch_idx >= 1:
            # 触发 on_exception；
            raise Exception
        print("on_train_epoch_begin")

    def on_train_epoch_end(self, trainer):
        print("on_train_epoch_end")

    def on_fetch_data_begin(self, trainer):
        print("on_fetch_data_begin")

    def on_fetch_data_end(self, trainer):
        print("on_fetch_data_end")

    def on_train_batch_begin(self, trainer, batch, indices=None):
        print("on_train_batch_begin")

    def on_train_batch_end(self, trainer):
        print("on_train_batch_end")

    def on_exception(self, trainer, exception):
        print("on_exception")

    def on_before_backward(self, trainer, outputs):
        print("on_before_backward")

    def on_after_backward(self, trainer):
        print("on_after_backward")

    def on_before_optimizers_step(self, trainer, optimizers):
        print("on_before_optimizers_step")

    def on_after_optimizers_step(self, trainer, optimizers):
        print("on_after_optimizers_step")

    def on_before_zero_grad(self, trainer, optimizers):
        print("on_before_zero_grad")

    def on_after_zero_grad(self, trainer, optimizers):
        print("on_after_zero_grad")

    def on_validate_begin(self, trainer):
        print("on_validate_begin")

    def on_validate_end(self, trainer, results):
        print("on_validate_end")









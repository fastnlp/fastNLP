import json
import os

from ...envs import _module_available, get_global_rank
from ..log import logger
from .has_monitor_callback import HasMonitorCallback

if _module_available('fitlog'):
    import fitlog

__all__ = ['FitlogCallback']


class FitlogCallback(HasMonitorCallback):
    r"""自动记录 ``evaluation`` 结果到 ``fitlog`` 中。会自动记录每次 ``evaluate``
    后的结果；同时会根据``monitor`` 记录最好的结果。另外，会自动将非 ``rank 0`` 上
    的 ``fitlog`` 设置为 ``debug`` 状态。同时还会在 ``fitlog`` 的 ``other`` 列中
    记录 一个 ``launch_time``，可以通过这个数值找到当前这个脚本的在 save_folder
    （如果有使用其它需要保存模型的 ``Callback``，例如 :class:`~fastNLP.core.
    callbacks.CheckpointCallback` ）下的文件夹名称。

    :param monitor: 监控的 metric 值。

        * 为 ``None`` 时，
          fastNLP 将尝试使用 :class:`~fastNLP.core.controllers.Trainer` 中设置
          的 `monitor` 值（如果有设置）。
        * 为 ``str`` 时，
          fastNLP 将尝试直接使用该名称从 ``evaluation`` 的结果中寻找，如果最终在
          ``evaluation`` 结果中没有找到完全一致的名称，则将使用最长公共字符串算法
          从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor``。
        * 为 :class:`Callable` 时，
          则接受参数为 ``evaluation`` 的结果（字典类型），返回一个 ``float`` 值作
          为 ``monitor`` 的结果，如果当前结果中没有相关的 ``monitor`` 值则返回
          ``None``。

    :param larger_better: 是否是越大越好。
    :param log_exception: 是否记录 ``exception``。
    :param log_loss_every: 多少个 ``batch`` 记录一次 loss 到 ``fitlog`` 中。
    """

    def __init__(self,
                 monitor=None,
                 larger_better: bool = True,
                 log_exception: bool = True,
                 log_loss_every: int = 0):
        assert _module_available('fitlog'), 'fitlog is not installed.'

        super().__init__(monitor=monitor, larger_better=larger_better)
        self.log_exception = log_exception
        self.log_loss_every = log_loss_every
        self.avg_loss = 0
        self.catch_exception = False

    def on_after_trainer_initialized(self, trainer, driver):
        if get_global_rank() != 0:
            # 如果不是 global rank 为 0 ，需要关闭 fitlog
            fitlog.debug()
        super().on_after_trainer_initialized(trainer, driver)
        fitlog.add_other(
            name='launch_time', value=os.environ['FASTNLP_LAUNCH_TIME'])
        if get_global_rank() == 0:
            # 这里主要为那种重新运行代码，需要延续之前的fitlog记录
            log_dir = fitlog.get_log_folder(absolute=True)
            if log_dir is not None:
                best_metric_fp = os.path.join(log_dir, 'best_metric.log')
                if os.path.exists(best_metric_fp):
                    results = {}
                    with open(best_metric_fp, 'r') as f:
                        for line in f:
                            _results = json.loads(line.strip())['metric']
                            results.update(_results)
                    if results:
                        self.is_better_results(results, keep_if_better=True)

    def on_sanity_check_end(self, trainer, sanity_check_res):
        super(FitlogCallback,
              self).on_sanity_check_end(trainer, sanity_check_res)
        if self.monitor is None:
            logger.rank_zero_warning(
                f'No monitor set for {self.log_name}. Therefore, no best '
                'metric will be logged.')

    def on_evaluate_end(self, trainer, results):
        results = self.itemize_results(results)
        fitlog.add_metric(
            results,
            step=trainer.global_forward_batches,
            epoch=trainer.cur_epoch_idx)
        if self.is_better_results(results, keep_if_better=True):
            results['step'] = trainer.global_forward_batches
            results['epoch'] = trainer.cur_epoch_idx
            fitlog.add_best_metric(results)

    def on_before_backward(self, trainer, outputs):
        if self.log_loss_every > 0:
            loss = trainer.extract_loss_from_outputs(outputs)
            self.avg_loss += loss.item()
            if trainer.global_forward_batches % self.log_loss_every == 0:
                fitlog.add_loss(
                    self.avg_loss / self.log_loss_every *
                    trainer.accumulation_steps,
                    name='loss',
                    step=trainer.global_forward_batches,
                    epoch=trainer.cur_epoch_idx)
                self.avg_loss = 0

    def on_train_end(self, trainer):
        if not self.catch_exception:
            fitlog.finish()

    def on_exception(self, trainer, exception):
        self.catch_exception = True
        fitlog.finish(status=1)
        if self.log_exception:
            fitlog.add_other(repr(exception), name='except_info')

import json
import sys
from typing import Union

__all__ = [
    'choose_progress_callback',
    'ProgressCallback',
    'RichCallback'
]

from .has_monitor_callback import HasMonitorCallback
from fastNLP.core.utils import f_rich_progress
from fastNLP.core.log import logger


class ProgressCallback(HasMonitorCallback):
    def on_train_end(self, trainer):
        f_rich_progress.stop()

    @property
    def name(self):  # progress bar的名称
        return 'auto'


def choose_progress_callback(progress_bar: Union[str, ProgressCallback]) -> ProgressCallback:
    if progress_bar == 'auto':
        if not f_rich_progress.dummy_rich:
            progress_bar = 'rich'
        else:
            progress_bar = 'raw'
    if progress_bar == 'rich':
        return RichCallback()
    elif progress_bar == 'raw':
        return RawTextCallback()
    elif isinstance(progress_bar, ProgressCallback):
        return progress_bar
    else:
        return None


class RichCallback(ProgressCallback):
    def __init__(self, print_every:int = 1, loss_round_ndigit:int = 6, monitor:str=None, larger_better:bool=True,
                 format_json=True):
        """

        :param print_every: 多少个 batch 更新一次显示。
        :param loss_round_ndigit: 显示的 loss 保留多少位有效数字
        :param monitor: 当检测到这个key的结果更好时，会打印出不同的颜色进行提示。监控的 metric 值。如果在 evaluation 结果中没有找到
            完全一致的名称，将使用 最长公共字符串算法 找到最匹配的那个作为 monitor 。如果为 None，将尝试使用 Trainer 设置的 monitor
            。也可以传入一个函数，接受参数为 evaluation 的结果(字典类型)，返回一个 float 值作为 monitor 的结果，如果当前结果中没有
            相关的 monitor 值请返回 None 。
        :param larger_better: 是否是 monitor 的结果越大越好。
        :param format_json: 是否格式化 json 再打印
        """
        super().__init__(monitor=monitor, larger_better=larger_better, must_have_monitor=False)
        self.print_every = print_every
        self.progress_bar = f_rich_progress
        self.task2id = {}
        self.loss = 0
        self.loss_round_ndigit = loss_round_ndigit
        self.format_json = format_json

    def on_after_trainer_initialized(self, trainer, driver):
        if not self.progress_bar.disable:
            self.progress_bar.set_disable(flag=trainer.driver.get_local_rank() != 0)
        super(RichCallback, self).on_after_trainer_initialized(trainer, driver)

    def on_train_begin(self, trainer):
        self.task2id['epoch'] = self.progress_bar.add_task(description='Epoch:0', total=trainer.n_epochs,
                                                    completed=trainer.global_forward_batches/(trainer.total_batches+1e-6))

    def on_train_epoch_begin(self, trainer):
        self.epoch_bar_update_advance = self.print_every/(trainer.num_batches_per_epoch + 1e-6)
        if 'batch' in self.task2id:
            self.progress_bar.reset(self.task2id['batch'], completed=trainer.batch_idx_in_epoch)
        else:
            self.task2id['batch'] = self.progress_bar.add_task(description='Batch:0', total=trainer.num_batches_per_epoch)

    def on_train_epoch_end(self, trainer):
        self.progress_bar.update(self.task2id['epoch'], description=f'Epoch:{trainer.cur_epoch_idx}',
                                 advance=None, completed=trainer.cur_epoch_idx, refresh=True)

    def on_train_end(self, trainer):
        self.clear_tasks()

    def on_before_backward(self, trainer, outputs):
        loss = trainer.extract_loss_from_outputs(outputs)
        loss = trainer.driver.tensor_to_numeric(loss, reduce='sum')
        self.loss += loss

    def on_train_batch_end(self, trainer):
        if trainer.global_forward_batches % self.print_every == 0:
            loss = self.loss/self.print_every
            self.loss = 0
            self.progress_bar.update(self.task2id['batch'], description=f'Batch:{trainer.batch_idx_in_epoch}',
                                     advance=self.print_every,
                                     post_desc=f'Loss:{round(loss, self.loss_round_ndigit)}', refresh=True)
            self.progress_bar.update(self.task2id['epoch'], description=f'Epoch:{trainer.cur_epoch_idx}',
                                     advance=self.epoch_bar_update_advance, refresh=True)

    def on_evaluate_end(self, trainer, results):
        if len(results)==0:
            return
        rule_style = ''
        text_style = ''
        characters = '-'
        if self.monitor is not None:
            monitor_value = self.get_monitor_value(results)
            if self.is_better_monitor_value(monitor_value, keep_if_better=True):
                if abs(self.monitor_value) != float('inf'):
                    rule_style = 'spring_green3'
                    text_style = '[bold]'
                    characters = '+'
        self.progress_bar.print()
        self.progress_bar.console.rule(text_style+f"Eval. results on Epoch:{trainer.cur_epoch_idx}, "
                                                  f"Batch:{trainer.batch_idx_in_epoch}",
                                       style=rule_style, characters=characters)
        if self.format_json:
            self.progress_bar.console.print_json(json.dumps(trainer.driver.tensor_to_numeric(results)))
        else:
            self.progress_bar.print(results)

    def clear_tasks(self):
        for key, taskid in self.task2id.items():
            self.progress_bar.destroy_task(taskid)
        self.progress_bar.stop()
        self.task2id = {}
        self.loss = 0

    @property
    def name(self):  # progress bar的名称
        return 'rich'


class RawTextCallback(ProgressCallback):
    def __init__(self, print_every:int = 1, loss_round_ndigit:int = 6, monitor:str=None, larger_better:bool=True,
                 format_json=True):
        """
        通过向命令行打印进度的方式显示

        :param print_every: 多少个 batch 更新一次显示。
        :param loss_round_ndigit: 显示的 loss 保留多少位有效数字
        :param monitor: 当检测到这个key的结果更好时，会打印出不同的颜色进行提示。监控的 metric 值。如果在 evaluation 结果中没有找到
            完全一致的名称，将使用 最长公共字符串算法 找到最匹配的那个作为 monitor 。如果为 None，将尝试使用 Trainer 设置的 monitor
            。也可以传入一个函数，接受参数为 evaluation 的结果(字典类型)，返回一个 float 值作为 monitor 的结果，如果当前结果中没有
            相关的 monitor 值请返回 None 。
        :param larger_better: 是否是monitor的结果越大越好。
        :param format_json: 是否format json再打印
        """
        super().__init__(monitor=monitor, larger_better=larger_better, must_have_monitor=False)
        self.print_every = print_every
        self.task2id = {}
        self.loss = 0
        self.loss_round_ndigit = loss_round_ndigit
        self.set_monitor(monitor, larger_better)
        self.format_json = format_json
        self.num_signs = 10

    def on_train_epoch_begin(self, trainer):
        logger.info('\n' + "*"*self.num_signs + f'Epoch:{trainer.cur_epoch_idx} starts' + '*'*self.num_signs)

    def on_before_backward(self, trainer, outputs):
        loss = trainer.extract_loss_from_outputs(outputs)
        loss = trainer.driver.tensor_to_numeric(loss, reduce='sum')
        self.loss += loss

    def on_train_batch_end(self, trainer):
        if trainer.global_forward_batches % self.print_every == 0:
            loss = self.loss/self.print_every
            self.loss = 0
            text = f'Epoch:{trainer.cur_epoch_idx}/{trainer.n_epochs}, Batch:{trainer.batch_idx_in_epoch}, ' \
                   f'loss:{round(loss, self.loss_round_ndigit)}, ' \
                   f'finished {round(trainer.global_forward_batches/trainer.total_batches*100, 2)}%.'
            logger.info(text)

    def on_evaluate_end(self, trainer, results):
        if len(results)==0:
            return
        base_text = f'Eval. results on Epoch:{trainer.cur_epoch_idx}, Batch:{trainer.batch_idx_in_epoch}'
        text = ''
        if self.monitor is not None:
            monitor_value = self.get_monitor_value(results)
            if self.is_better_monitor_value(monitor_value, keep_if_better=True):
                if abs(self.monitor_value) != float('inf'):
                    text = '+'*self.num_signs + base_text + '+'*self.num_signs
        if len(text) == 0:
            text = '-'*self.num_signs + base_text + '-'*self.num_signs

        logger.info(text)
        if self.format_json:
            logger.info(json.dumps(trainer.driver.tensor_to_numeric(results)))
        else:
            logger.info(results)

    @property
    def name(self):  # progress bar的名称
        return 'raw'
import json
from typing import Union

__all__ = [
    'choose_progress_callback',
    'ProgressCallback',
    'RichCallback',
    'TqdmCallback'
]


from .has_monitor_callback import HasMonitorCallback
from fastNLP.core.utils import f_rich_progress, f_tqdm_progress
from fastNLP.core.log import logger


class ProgressCallback(HasMonitorCallback):
    def __init__(self, monitor, larger_better, must_have_monitor=False):
        super(ProgressCallback, self).__init__(monitor=monitor, larger_better=larger_better,
                                               must_have_monitor=must_have_monitor)
        self.best_monitor_epoch = -1
        self.best_monitor_step = -1
        self.best_results = None

    def record_better_monitor(self, trainer):
        self.best_monitor_step = trainer.global_forward_batches
        self.best_monitor_epoch = trainer.cur_epoch_idx

    def on_train_end(self, trainer):
        if self.best_monitor_epoch != -1:
            msg = f"The best performance for monitor {self._real_monitor}:{self.monitor_value} was achieved in" \
                  f" Epoch:{self.best_monitor_epoch}, Global Batch:{self.best_monitor_step}."
            if self.best_results is not None:
                msg = msg + ' The evaluation result: \n' + str(self.best_results)
            logger.info(msg)

    @property
    def name(self):  # progress bar的名称
        return 'auto'


def choose_progress_callback(progress_bar: Union[str, ProgressCallback]) -> ProgressCallback:
    if progress_bar == 'auto':
        if not f_rich_progress.dummy:
            progress_bar = 'rich'
        else:
            progress_bar = 'raw'
    if progress_bar == 'rich':
        return RichCallback()
    elif progress_bar == 'raw':
        return RawTextCallback()
    elif progress_bar == 'tqdm':
        return TqdmCallback()
    elif isinstance(progress_bar, ProgressCallback):
        return progress_bar
    else:
        return None


class RichCallback(ProgressCallback):
    """
    在训练过程中打印 rich progress bar 的 callback 。在 Trainer 中，默认就会使用这个 callback 来显示进度。如果需要定制这个 Callback 的
    参数，请通过实例化本 Callback 并传入到 Trainer 中实现。在打印 evaluate 的结果时，不会打印名称以 "_" 开头的内容。

    :param print_every: 多少个 batch 更新一次显示。
    :param loss_round_ndigit: 显示的 loss 保留多少位有效数字
    :param monitor: 监控的 metric 值。当检测到这个key的结果更好时，会打印出不同的颜色进行提示。

        * 为 ``None``
         将尝试使用 :class:`~fastNLP.Trainer` 中设置 `monitor` 值（如果有设置）。
        * 为 ``str``
         尝试直接使用该名称从 ``evaluation`` 结果中寻找，如果在 ``evaluation`` 结果中没有找到完全一致的名称，将
         使用 最长公共字符串算法 从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor`` 。
        * 为 ``Callable``
         接受参数为 ``evaluation`` 的结果(字典类型)，返回一个 ``float`` 值作为 ``monitor`` 的结果，如果当前结果中没有相关
         的 ``monitor`` 值请返回 ``None`` 。
    :param larger_better: 是否是 monitor 的结果越大越好。
    :param format_json: 是否格式化 json 再打印
    """
    def __init__(self, print_every:int = 1, loss_round_ndigit:int = 6, monitor:str=None, larger_better:bool=True,
                 format_json=True):
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
        self.task2id['epoch'] = self.progress_bar.add_task(description=f'Epoch:{trainer.cur_epoch_idx}',
                                                           total=trainer.n_epochs,
                                                           completed=trainer.global_forward_batches/(trainer.n_batches+1e-6)*
                                                           trainer.n_epochs)

    def on_train_epoch_begin(self, trainer):
        self.epoch_bar_update_advance = self.print_every/(trainer.num_batches_per_epoch + 1e-6)
        if 'batch' in self.task2id:
            self.progress_bar.reset(self.task2id['batch'], completed=trainer.batch_idx_in_epoch)
        else:
            self.task2id['batch'] = self.progress_bar.add_task(description=f'Batch:{trainer.batch_idx_in_epoch}',
                                                               total=trainer.num_batches_per_epoch,
                                                               completed=trainer.batch_idx_in_epoch)

    def on_train_epoch_end(self, trainer):
        self.progress_bar.update(self.task2id['epoch'], description=f'Epoch:{trainer.cur_epoch_idx}',
                                 advance=None, completed=trainer.cur_epoch_idx, refresh=True)

    def on_train_end(self, trainer):
        super(RichCallback, self).on_train_end(trainer)
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
            if self.is_better_results(results, keep_if_better=True):
                self.record_better_monitor(trainer)
                if abs(self.monitor_value) != float('inf'):
                    rule_style = 'spring_green3'
                    text_style = '[bold]'
                    characters = '+'
        self.progress_bar.print()
        self.progress_bar.console.rule(text_style+f"Eval. results on Epoch:{trainer.cur_epoch_idx}, "
                                                  f"Batch:{trainer.batch_idx_in_epoch}",
                                       style=rule_style, characters=characters)
        results = {key:trainer.driver.tensor_to_numeric(value) for key, value in results.items() if
                   not key.startswith('_')}
        if self.format_json:
            results = json.dumps(results)
            self.progress_bar.console.print_json(results)
        else:
            self.progress_bar.print(results)
        self.best_results = results

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
        通过向命令行打印进度的方式显示。在打印 evaluate 的结果时，不会打印名称以 "_" 开头的内容。

        :param print_every: 多少个 batch 更新一次显示。
        :param loss_round_ndigit: 显示的 loss 保留多少位有效数字
        :param monitor: 监控的 metric 值。当检测到这个key的结果更好时，会打印出不同的颜色进行提示。

            * 为 ``None``
             将尝试使用 :class:`~fastNLP.Trainer` 中设置 `monitor` 值（如果有设置）。
            * 为 ``str``
             尝试直接使用该名称从 ``evaluation`` 结果中寻找，如果在 ``evaluation`` 结果中没有找到完全一致的名称，将
             使用 最长公共字符串算法 从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor`` 。
            * 为 ``Callable``
             接受参数为 ``evaluation`` 的结果(字典类型)，返回一个 ``float`` 值作为 ``monitor`` 的结果，如果当前结果中没有相关
             的 ``monitor`` 值请返回 ``None`` 。
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
                   f'finished {round(trainer.global_forward_batches/trainer.n_batches*100, 2)}%.'
            logger.info(text)

    def on_evaluate_end(self, trainer, results):
        if len(results)==0:
            return
        base_text = f'Eval. results on Epoch:{trainer.cur_epoch_idx}, Batch:{trainer.batch_idx_in_epoch}'
        text = ''
        if self.monitor is not None:
            if self.is_better_results(results, keep_if_better=True):
                self.record_better_monitor(trainer)
                if abs(self.monitor_value) != float('inf'):
                    text = '+'*self.num_signs + base_text + '+'*self.num_signs
        if len(text) == 0:
            text = '-'*self.num_signs + base_text + '-'*self.num_signs

        logger.info(text)
        results = {key:trainer.driver.tensor_to_numeric(value) for key, value in results.items() if
                   not key.startswith('_')}
        if self.format_json:
            results = json.dumps(results)
        logger.info(results)
        self.best_results = results

    @property
    def name(self):  # progress bar的名称
        return 'raw'


class TqdmCallback(ProgressCallback):
    """
    在训练过程中打印 tqdm progress bar 的 callback 。在 Trainer 中，默认就会使用这个 callback 来显示进度。如果需要定制这个 Callback 的
    参数，请通过实例化本 Callback 并传入到 Trainer 中实现。在打印 evaluate 的结果时，不会打印名称以 "_" 开头的内容。

    :param print_every: 多少个 batch 更新一次显示。
    :param loss_round_ndigit: 显示的 loss 保留多少位有效数字
    :param monitor: 监控的 metric 值。当检测到这个key的结果更好时，会打印出不同的颜色进行提示。

        * 为 ``None``
         将尝试使用 :class:`~fastNLP.Trainer` 中设置 `monitor` 值（如果有设置）。
        * 为 ``str``
         尝试直接使用该名称从 ``evaluation`` 结果中寻找，如果在 ``evaluation`` 结果中没有找到完全一致的名称，将
         使用 最长公共字符串算法 从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor`` 。
        * 为 ``Callable``
         接受参数为 ``evaluation`` 的结果(字典类型)，返回一个 ``float`` 值作为 ``monitor`` 的结果，如果当前结果中没有相关
         的 ``monitor`` 值请返回 ``None`` 。
    :param larger_better: 是否是 monitor 的结果越大越好。
    :param format_json: 是否格式化 json 再打印
    """
    def __init__(self, print_every:int = 1, loss_round_ndigit:int = 6, monitor:str=None, larger_better:bool=True,
                 format_json=True):
        super().__init__(monitor=monitor, larger_better=larger_better, must_have_monitor=False)
        self.print_every = print_every
        self.progress_bar = f_tqdm_progress
        self.task2id = {}
        self.loss = 0
        self.loss_round_ndigit = loss_round_ndigit
        self.format_json = format_json
        self.num_signs = 10

    def on_train_begin(self, trainer):
        self.task2id['epoch'] = self.progress_bar.add_task(description=f'Epoch:{trainer.cur_epoch_idx}',
                                                           total=trainer.n_epochs,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}, {postfix}]',
            initial=trainer.global_forward_batches/(trainer.n_batches+1e-6)*trainer.n_epochs)

    def on_train_epoch_begin(self, trainer):
        self.epoch_bar_update_advance = self.print_every/(trainer.num_batches_per_epoch + 1e-6)
        if 'batch' in self.task2id:
            self.progress_bar.reset(self.task2id['batch'])
        else:
            self.task2id['batch'] = self.progress_bar.add_task(description='Batch', total=trainer.num_batches_per_epoch,
                                                               initial=trainer.batch_idx_in_epoch)
        self.progress_bar.set_description_str(self.task2id['epoch'], f'Epoch:{trainer.cur_epoch_idx}', refresh=True)

    def on_train_end(self, trainer):
        super(TqdmCallback, self).on_train_end(trainer)
        self.clear_tasks()

    def on_before_backward(self, trainer, outputs):
        loss = trainer.extract_loss_from_outputs(outputs)
        loss = trainer.driver.tensor_to_numeric(loss, reduce='sum')
        self.loss += loss

    def on_train_batch_end(self, trainer):
        if trainer.global_forward_batches % self.print_every == 0:
            loss = self.loss/self.print_every
            self.loss = 0
            self.progress_bar.update(self.task2id['batch'], advance=self.print_every, refresh=True)
            self.progress_bar.set_postfix_str(self.task2id['batch'], f'Loss:{round(loss, self.loss_round_ndigit)}')
            self.progress_bar.update(self.task2id['epoch'], advance=self.epoch_bar_update_advance, refresh=True)

    def on_evaluate_end(self, trainer, results):
        if len(results) == 0:
            return
        base_text = f'Eval. results on Epoch:{trainer.cur_epoch_idx}, Batch:{trainer.batch_idx_in_epoch}'
        text = ''
        if self.monitor is not None:
            if self.is_better_results(results, keep_if_better=True):
                self.record_better_monitor(trainer)
                if abs(self.monitor_value) != float('inf'):
                    text = '+'*self.num_signs + base_text + '+'*self.num_signs
        if len(text) == 0:
            text = '-'*self.num_signs + base_text + '-'*self.num_signs

        logger.info(text)
        results = {key:trainer.driver.tensor_to_numeric(value) for key, value in results.items() if
                   not key.startswith('_')}
        if self.format_json:
            results = json.dumps(results)
        logger.info(results)
        self.best_results = results

    def clear_tasks(self):
        for key, taskid in self.task2id.items():
            self.progress_bar.destroy_task(taskid)
        self.task2id = {}
        self.loss = 0

    @property
    def name(self):  # progress bar的名称
        return 'tqdm'

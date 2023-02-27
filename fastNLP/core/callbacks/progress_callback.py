import contextlib
import json
import os
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Union

from fastNLP.core.log import logger
from fastNLP.core.utils import f_rich_progress, f_tqdm_progress
from .has_monitor_callback import HasMonitorCallback

__all__ = [
    'choose_progress_callback', 'ProgressCallback', 'RichCallback',
    'TqdmCallback', 'RawTextCallback', 'ExtraInfoStatistics'
]


class ExtraInfoStatistics:
    """用于记录要显示的额外信息的统计工具类。

    :param extra_show_keys: 要显示的额外信息的key

        * 为 ``str``
         从 train_step 的返回 的 dict 中寻找该名称的内容，如果找到则打印出来。
        * 为 ``Sequence[str]``
         与 ``str`` 类似，但是可以打印多个内容。
    """

    @abstractmethod
    def update(self, outputs: dict) -> None:
        r"""
        每个 batch 训练结束后调用。

        :param outputs: 一般是直接来自于模型的输出或经过 output_mapping 的结果。

        """

    @abstractmethod
    def get_stat(self) -> Dict:
        r"""
        每个打印周期调用一次，返回的字典中的内容将被显示到进度条。
        """


class DefaultExtraInfoStatistics(ExtraInfoStatistics):
    """用于记录要显示的额外信息的统计工具类。

    :param extra_show_keys: 要显示的额外信息的key

        * 为 ``str``
         从 train_step 的返回 的 dict 中寻找该名称的内容，如果找到则打印出来。
        * 为 ``Sequence[str]``
         与 ``str`` 类似，但是可以打印多个内容。
    """

    def __init__(self,
                 extra_show_keys: Union[str, Sequence[str]],
                 round_ndigit=6):
        self.extra_show_keys = [extra_show_keys] if isinstance(
            extra_show_keys, str) else extra_show_keys
        self.extra_info_collection: Dict[str, float] = {}
        self.extra_info_collection_counter: Dict[str, int] = defaultdict(int)
        self.round_ndigit = round_ndigit
        self.driver = None

    def update(self, outputs: dict) -> None:
        r"""
        每个 batch 训练结束后调用。

        :param outputs: 一般是直接来自于模型的输出或经过 output_mapping 的结果。

        """
        for k in self.extra_show_keys:
            if k not in outputs:
                continue
            v = outputs[k]
            if self.driver is not None:
                v = self.driver.tensor_to_numeric(v, reduce='sum')
            if k in self.extra_info_collection:
                self.extra_info_collection[k] += v
            else:
                self.extra_info_collection[k] = v
            self.extra_info_collection_counter[k] += 1

    def get_stat(self) -> Dict:
        r"""
        每个打印周期调用一次，返回的字典中的内容将被显示到进度条。
        """
        res = {}
        for k, v in self.extra_info_collection.items():
            v = v / self.extra_info_collection_counter[k]
            res[k] = round(v, self.round_ndigit)
        self.extra_info_collection.clear()
        self.extra_info_collection_counter.clear()
        return res

    def set_driver(self, driver):
        """设置 Driver 对象，用于方便将 tensor 类型的数据转为数字。

        :param driver:
        :return:
        """
        self.driver = driver


def _get_beautiful_extra_string(extra_info_collection: dict,
                                progress_type: str) -> str:
    """用户有额外信息需要打印的情况下，将额外信息转换为字符串。"""
    original = ''.join(
        [f', {key}:{value}' for key, value in extra_info_collection.items()])
    if progress_type == 'rich':
        try:
            import os
            width = os.get_terminal_size().columns
            if width - 60 < len(original):
                # 60 是为进度条预留的宽度，此时较为美观。
                original = original.replace(',', ',\n')
        except OSError:
            # 某些特殊情况，系统并不支持获取控制台宽度的
            pass
    elif progress_type == 'tqdm':
        # TODO 使用 TQDM 进度条时，额外信息过长，控制台太小时会导致额外信息显示不
        # 全，且无法和 RICH 一样通过换行规避。
        pass
    return original


class ProgressCallback(HasMonitorCallback):
    r"""在 **fastNLP** 中显示进度条的 ``Callback``。一般作为具体 progress_bar 的父
    类被继承。

    :param monitor: 监控的 metric 值。当检测到这个key的结果更好时，会打印出不同的颜
        色进行提示。

        * 为 ``None`` 时，
          fastNLP 将尝试使用 :class:`.Trainer` 中设置的 `monitor` 值（如果有设
          置）。
        * 为 ``str`` 时，
          fastNLP 将尝试直接使用该名称从 ``evaluation`` 的结果中寻找，如果最终在
          ``evaluation`` 结果中没有找到完全一致的名称，则将使用最长公共字符串算法
          从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor``。
        * 为 :class:`Callable` 时，
          则接受参数为 ``evaluation`` 的结果（字典类型），返回一个 ``float`` 值作
          为 ``monitor`` 的结果，如果当前结果中没有相关的 ``monitor`` 值则返回
          ``None``。

    :param larger_better: 是否是 monitor 的结果越大越好。
    :param must_have_monitor: 是否强制要求传入 ``monitor``。
    """

    def __init__(self, monitor, larger_better, must_have_monitor=False):
        super(ProgressCallback, self).__init__(
            monitor=monitor,
            larger_better=larger_better,
            must_have_monitor=must_have_monitor)
        self.best_monitor_epoch = -1
        self.best_monitor_step = -1
        self.best_results = None

    def record_better_monitor(self, trainer, results):
        self.best_monitor_step = trainer.global_forward_batches
        self.best_monitor_epoch = trainer.cur_epoch_idx
        self.best_results = self.itemize_results(results)

    def on_train_end(self, trainer):
        if self.best_monitor_epoch != -1:
            msg = f'The best performance for monitor {self._real_monitor}: ' \
                  f'{self.monitor_value} was achieved in ' \
                  f'Epoch:{self.best_monitor_epoch}, ' \
                  f'Global Batch:{self.best_monitor_step}.'
            if self.best_results is not None:
                msg = msg + ' The evaluation result: \n' + str(
                    self.best_results)
            logger.info(msg)

    @property
    def name(self):  # progress bar的名称
        return 'auto'


def choose_progress_callback(
    progress_bar: Union[str, ProgressCallback],
    extra_show_keys: Union[str, Sequence[str], ExtraInfoStatistics, None]
) -> Optional[ProgressCallback]:
    if progress_bar == 'auto':
        if not f_rich_progress.dummy:
            progress_bar = 'rich'
        else:
            progress_bar = 'raw'
    if progress_bar == 'rich':
        return RichCallback(extra_show_keys=extra_show_keys)
    elif progress_bar == 'raw':
        return RawTextCallback(extra_show_keys=extra_show_keys)
    elif progress_bar == 'tqdm':
        return TqdmCallback(extra_show_keys=extra_show_keys)
    elif isinstance(progress_bar, ProgressCallback):
        return progress_bar
    else:
        return None


class RichCallback(ProgressCallback):
    r"""在训练过程中打印 *rich* progress bar 的 ``Callback``，也是在 :class:`.\
    Trainer` 中默认使用的进度条。

    这个 callback 来显示进度。如果需要定制这个 Callback 的 参数，请通过实例化本
    Callback 并传入到 :class:`.Trainer` 中实现。在打印 evaluate 的结果时，不会打印
    名称以 "_"开头的内容。

    :param print_every: 多少个 batch 更新一次显示。
    :param loss_round_ndigit: 显示的 loss 保留多少位有效数字
    :param monitor: 监控的 metric 值。当检测到这个key的结果更好时，会打印出不同的颜
        色进行提示。

        * 为 ``None`` 时，
          fastNLP 将尝试使用 :class:`.Trainer` 中设置的 `monitor` 值（如果有设
          置）。
        * 为 ``str`` 时，
          fastNLP 将尝试直接使用该名称从 ``evaluation`` 的结果中寻找，如果最终在
          ``evaluation`` 结果中没有找到完全一致的名称，则将使用最长公共字符串算法
          从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor``。
        * 为 :class:`Callable` 时，
          则接受参数为 ``evaluation`` 的结果（字典类型），返回一个 ``float`` 值作
          为 ``monitor`` 的结果，如果当前结果中没有相关的 ``monitor`` 值则返回
          ``None``。

    :param larger_better: 是否是 monitor 的结果越大越好。
    :param format_json: 是否格式化 json 再打印
    :param extra_show_keys: 每个 batch 训练结束后除了 loss 以外需要额外显示的内
        容。

        * 为 ``str`` 时，从 train_step 的返回 的 dict 中寻找该名称的内容，如果找到
          则打印出来。
        * 为 ``Sequence[str]`` 时，与 ``str`` 类似，但是可以打印多个内容。
        * 为 ``None`` 时，进度条不进行额外信息的展示。
        * 为 :class:`ExtraInfoStatistics` 及其子类时，必须实现 ``update`` 和
          ``get_stat`` 方法，在未进行打印的轮次，将额外信息相加累积，并在输出前求平
          均。可以手动重写 ``update`` 和 ``get_stat`` 方法来实现自己的累积平均方
          式。
    """

    def __init__(self,
                 print_every: int = 1,
                 loss_round_ndigit: int = 6,
                 monitor: Optional[str] = None,
                 larger_better: bool = True,
                 format_json=True,
                 extra_show_keys: Union[str, Sequence[str],
                                        ExtraInfoStatistics, None] = None):
        super().__init__(
            monitor=monitor,
            larger_better=larger_better,
            must_have_monitor=False)
        self.print_every = print_every
        self.progress_bar = f_rich_progress
        self.task2id: Dict[str, Any] = {}
        self.loss = 0
        self.loss_round_ndigit = loss_round_ndigit
        self.format_json = format_json
        # 用于在进度条中显示额外信息
        self.extra_show_keys = None
        if isinstance(extra_show_keys, ExtraInfoStatistics):
            self.extra_show_keys = extra_show_keys
        elif isinstance(extra_show_keys, str) or isinstance(
                extra_show_keys, Sequence):
            self.extra_show_keys = DefaultExtraInfoStatistics(
                extra_show_keys, round_ndigit=self.loss_round_ndigit)

    def on_after_trainer_initialized(self, trainer, driver):
        if not self.progress_bar.disable:
            self.progress_bar.set_disable(
                flag=trainer.driver.get_local_rank() != 0)
        super(RichCallback, self).on_after_trainer_initialized(trainer, driver)
        if isinstance(self.extra_show_keys, DefaultExtraInfoStatistics):
            self.extra_show_keys.set_driver(driver)

    def on_train_begin(self, trainer):
        self.task2id['epoch'] = self.progress_bar.add_task(
            description=f'Epoch:{trainer.cur_epoch_idx}',
            total=trainer.n_epochs,
            completed=trainer.global_forward_batches /
            (trainer.n_batches + 1e-6) * trainer.n_epochs)

    def on_train_epoch_begin(self, trainer):
        self.epoch_bar_update_advance = self.print_every / (
            trainer.num_batches_per_epoch + 1e-6)
        if 'batch' in self.task2id:
            self.progress_bar.reset(
                self.task2id['batch'], completed=trainer.batch_idx_in_epoch)
        else:
            self.task2id['batch'] = self.progress_bar.add_task(
                description=f'Batch:{trainer.batch_idx_in_epoch}',
                total=trainer.num_batches_per_epoch,
                completed=trainer.batch_idx_in_epoch)

    def on_train_epoch_end(self, trainer):
        self.progress_bar.update(
            self.task2id['epoch'],
            description=f'Epoch:{trainer.cur_epoch_idx}',
            advance=None,
            completed=trainer.cur_epoch_idx,
            refresh=True)

    def on_train_end(self, trainer):
        super(RichCallback, self).on_train_end(trainer)
        self.clear_tasks()

    def on_before_backward(self, trainer, outputs):
        loss = trainer.extract_loss_from_outputs(outputs)
        loss = trainer.driver.tensor_to_numeric(loss, reduce='sum')
        self.loss += loss

        # 如果有额外的信息需要显示
        if self.extra_show_keys is not None:
            self.extra_show_keys.update(outputs)

    def on_train_batch_end(self, trainer):
        if trainer.global_forward_batches % self.print_every == 0:
            loss = self.loss / self.print_every
            self.loss = 0
            # 默认情况下进度条后只有 Loss 信息
            post_desc = f'Loss:{loss:.{self.loss_round_ndigit}f}'
            # 进度条后附加上用户希望的额外信息
            if self.extra_show_keys is not None:
                post_desc = post_desc + \
                    _get_beautiful_extra_string(
                        self.extra_show_keys.get_stat(), progress_type='rich')
            self.progress_bar.update(
                self.task2id['batch'],
                description=f'Batch:{trainer.batch_idx_in_epoch}',
                advance=self.print_every,
                post_desc=post_desc,
                refresh=True)
            self.progress_bar.update(
                self.task2id['epoch'],
                description=f'Epoch:{trainer.cur_epoch_idx}',
                advance=self.epoch_bar_update_advance,
                refresh=True)

    def on_evaluate_end(self, trainer, results):
        if len(results) == 0:
            return
        rule_style = ''
        text_style = ''
        characters = '-'
        if self.monitor is not None:
            if self.is_better_results(results, keep_if_better=True):
                self.record_better_monitor(trainer, results)
                if abs(self.monitor_value) != float('inf'):
                    rule_style = 'spring_green3'
                    text_style = '[bold]'
                    characters = '+'
        self.progress_bar.print()
        text = f'Eval. results on Epoch:{trainer.cur_epoch_idx}, ' \
               f'Batch:{trainer.batch_idx_in_epoch}'
        self.progress_bar.console.rule(
            text_style + text, style=rule_style, characters=characters)
        with open(os.devnull, 'w') as f:
            # 这样可以让logger打印到文件记录中，但是不要再次输出到terminal中
            with contextlib.redirect_stdout(f):
                logger.info(f'{characters}' * 10 + text + f'{characters}' * 10)
        results = {
            key: trainer.driver.tensor_to_numeric(value)
            for key, value in results.items() if not key.startswith('_')
        }
        if self.format_json:
            results = json.dumps(results, indent=2)
            # self.progress_bar.console.print_json(results)
        # else:
        #     self.progress_bar.print(results)
        logger.info(results)

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
    r"""在命令行中以文本形式输出进度的 ``Callback``。

    通过向命令行打印进度的方式显示。在打印 evaluate 的结果时，不会打印名称以 "_"
    开头的内容。

    :param print_every: 多少个 batch 更新一次显示。
    :param loss_round_ndigit: 显示的 loss 保留多少位有效数字
    :param monitor: 监控的 metric 值。当检测到这个key的结果更好时，会打印出不同的颜
        色进行提示。

        * 为 ``None`` 时，
          fastNLP 将尝试使用 :class:`.Trainer` 中设置的 `monitor` 值（如果有设
          置）。
        * 为 ``str`` 时，
          fastNLP 将尝试直接使用该名称从 ``evaluation`` 的结果中寻找，如果最终在
          ``evaluation`` 结果中没有找到完全一致的名称，则将使用最长公共字符串算法
          从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor``。
        * 为 :class:`Callable` 时，
          则接受参数为 ``evaluation`` 的结果（字典类型），返回一个 ``float`` 值作
          为 ``monitor`` 的结果，如果当前结果中没有相关的 ``monitor`` 值则返回
          ``None``。
    :param larger_better: 是否是monitor的结果越大越好。
    :param format_json: 是否format json再打印。
    :param extra_show_keys: 每个 batch 训练结束后除了 loss 以外需要额外显示的内
        容。

        * 为 ``str`` 时，从 train_step 的返回 的 dict 中寻找该名称的内容，如果找到
          则打印出来。
        * 为 ``Sequence[str]`` 时，与 ``str`` 类似，但是可以打印多个内容。
        * 为 ``None`` 时，进度条不进行额外信息的展示。
        * 为 :class:`ExtraInfoStatistics` 及其子类时，必须实现 ``update`` 和
          ``get_stat`` 方法，在未进行打印的轮次，将额外信息相加累积，并在输出前求平
          均。可以手动重写 ``update`` 和 ``get_stat`` 方法来实现自己的累积平均方
          式。
    """

    def __init__(self,
                 print_every: int = 1,
                 loss_round_ndigit: int = 6,
                 monitor: Optional[str] = None,
                 larger_better: bool = True,
                 format_json=True,
                 extra_show_keys: Union[str, Sequence[str],
                                        ExtraInfoStatistics, None] = None):
        super().__init__(
            monitor=monitor,
            larger_better=larger_better,
            must_have_monitor=False)
        self.print_every = print_every
        self.task2id: Dict[str, Any] = {}
        self.loss = 0
        self.loss_round_ndigit = loss_round_ndigit
        self.set_monitor(monitor, larger_better)
        self.format_json = format_json
        self.num_signs = 10
        # 用于在进度条中显示额外信息
        self.extra_show_keys = None
        if isinstance(extra_show_keys, ExtraInfoStatistics):
            self.extra_show_keys = extra_show_keys
        elif isinstance(extra_show_keys, str) or isinstance(
                extra_show_keys, Sequence):
            self.extra_show_keys = DefaultExtraInfoStatistics(
                extra_show_keys, round_ndigit=self.loss_round_ndigit)

    def on_after_trainer_initialized(self, trainer, driver):
        if isinstance(self.extra_show_keys, DefaultExtraInfoStatistics):
            self.extra_show_keys.set_driver(driver)

    def on_train_epoch_begin(self, trainer):
        logger.info('\n' + '*' * self.num_signs +
                    f'Epoch:{trainer.cur_epoch_idx} starts' +
                    '*' * self.num_signs)

    def on_before_backward(self, trainer, outputs):
        loss = trainer.extract_loss_from_outputs(outputs)
        loss = trainer.driver.tensor_to_numeric(loss, reduce='sum')
        self.loss += loss

        # 如果有额外的信息需要显示
        if self.extra_show_keys is not None:
            self.extra_show_keys.update(outputs)

    def on_train_batch_end(self, trainer):
        if trainer.global_forward_batches % self.print_every == 0:
            loss = self.loss / self.print_every
            self.loss = 0
            # 默认情况下进度条后只有 Loss 信息
            post_desc = f'Loss:{loss:.{self.loss_round_ndigit}f}'
            # 进度条后附加上用户希望的额外信息
            if self.extra_show_keys is not None:
                post_desc = post_desc + \
                    _get_beautiful_extra_string(
                        self.extra_show_keys.get_stat(), progress_type='raw')
            percentage = round(
                trainer.global_forward_batches / trainer.n_batches * 100, 2)
            text = f'Epoch:{trainer.cur_epoch_idx}/{trainer.n_epochs}, ' \
                   f'Batch:{trainer.batch_idx_in_epoch}, ' + post_desc + \
                   f', finished {percentage}%.'
            logger.info(text)

    def on_evaluate_end(self, trainer, results):
        if len(results) == 0:
            return
        base_text = f'Eval. results on Epoch:{trainer.cur_epoch_idx}, ' \
                    f'Batch:{trainer.batch_idx_in_epoch}'
        text = ''
        if self.monitor is not None:
            if self.is_better_results(results, keep_if_better=True):
                self.record_better_monitor(trainer, results)
                if abs(self.monitor_value) != float('inf'):
                    text = '+' * self.num_signs + base_text + \
                           '+' * self.num_signs
        if len(text) == 0:
            text = '-' * self.num_signs + base_text + '-' * self.num_signs

        logger.info(text)
        results = {
            key: trainer.driver.tensor_to_numeric(value)
            for key, value in results.items() if not key.startswith('_')
        }
        if self.format_json:
            results = json.dumps(results, indent=2)
        logger.info(results)

    @property
    def name(self):  # progress bar的名称
        return 'raw'


class TqdmCallback(ProgressCallback):
    r"""在训练过程中打印 *tqdm* progress bar 的 ``Callback``。

    在 :class:`.Trainer` 中，如果设置了 ``progress_bar='tqdm'`` 就会使用
    ``TqdmCallback`` 来显示进度。如果需要定制这个 Callback 的参数，请通过实例化本
    Callback 并传入到 Trainer 中实现。同时在打印 evaluate 的结果时，不会打印名称以
    "_" 开头的内容。

    :param print_every: 多少个 batch 更新一次显示。
    :param loss_round_ndigit: 显示的 loss 保留多少位有效数字
    :param monitor: 监控的 metric 值。当检测到这个key的结果更好时，会打印出不同的颜
        色进行提示。

        * 为 ``None`` 时，
          fastNLP 将尝试使用 :class:`.Trainer` 中设置的 `monitor` 值（如果有设
          置）。
        * 为 ``str`` 时，
          fastNLP 将尝试直接使用该名称从 ``evaluation`` 的结果中寻找，如果最终在
          ``evaluation`` 结果中没有找到完全一致的名称，则将使用最长公共字符串算法
          从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor``。
        * 为 :class:`Callable` 时，
          则接受参数为 ``evaluation`` 的结果（字典类型），返回一个 ``float`` 值作
          为 ``monitor`` 的结果，如果当前结果中没有相关的 ``monitor`` 值则返回
          ``None``。
    :param larger_better: 是否是 monitor 的结果越大越好。
    :param format_json: 是否格式化 json 再打印
    :param extra_show_keys: 每个 batch 训练结束后除了 loss 以外需要额外显示的内
        容。

        * 为 ``str`` 时，从 train_step 的返回 的 dict 中寻找该名称的内容，如果找到
          则打印出来。
        * 为 ``Sequence[str]`` 时，与 ``str`` 类似，但是可以打印多个内容。
        * 为 ``None`` 时，进度条不进行额外信息的展示。
        * 为 :class:`ExtraInfoStatistics` 及其子类时，必须实现 ``update`` 和
          ``get_stat`` 方法，在未进行打印的轮次，将额外信息相加累积，并在输出前求平
          均。可以手动重写 ``update`` 和 ``get_stat`` 方法来实现自己的累积平均方
          式。
    """

    def __init__(self,
                 print_every: int = 1,
                 loss_round_ndigit: int = 6,
                 monitor: Optional[str] = None,
                 larger_better: bool = True,
                 format_json=True,
                 extra_show_keys: Union[str, Sequence[str],
                                        ExtraInfoStatistics, None] = None):
        super().__init__(
            monitor=monitor,
            larger_better=larger_better,
            must_have_monitor=False)
        self.print_every = print_every
        self.progress_bar = f_tqdm_progress
        self.task2id: Dict[str, Any] = {}
        self.loss = 0
        self.loss_round_ndigit = loss_round_ndigit
        self.format_json = format_json
        self.num_signs = 10
        # 用于在进度条中显示额外信息
        self.extra_show_keys = None
        if isinstance(extra_show_keys, ExtraInfoStatistics):
            self.extra_show_keys = extra_show_keys
        elif isinstance(extra_show_keys, str) or isinstance(
                extra_show_keys, Sequence):
            self.extra_show_keys = DefaultExtraInfoStatistics(
                extra_show_keys, round_ndigit=self.loss_round_ndigit)

    def on_after_trainer_initialized(self, trainer, driver):
        if isinstance(self.extra_show_keys, DefaultExtraInfoStatistics):
            self.extra_show_keys.set_driver(driver)

    def on_train_begin(self, trainer):
        self.task2id['epoch'] = self.progress_bar.add_task(
            description=f'Epoch:{trainer.cur_epoch_idx}',
            total=trainer.n_epochs,
            dynamic_ncols=True,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| '
            '[{elapsed}<{remaining}, {rate_fmt}, {postfix}]',
            initial=trainer.global_forward_batches /
            (trainer.n_batches + 1e-6) * trainer.n_epochs)

    def on_train_epoch_begin(self, trainer):
        self.epoch_bar_update_advance = self.print_every / (
            trainer.num_batches_per_epoch + 1e-6)
        if 'batch' in self.task2id:
            self.progress_bar.reset(self.task2id['batch'])
        else:
            self.task2id['batch'] = self.progress_bar.add_task(
                description='Batch',
                total=trainer.num_batches_per_epoch,
                initial=trainer.batch_idx_in_epoch)
        self.progress_bar.set_description_str(
            self.task2id['epoch'],
            f'Epoch:{trainer.cur_epoch_idx}',
            refresh=True)

    def on_train_end(self, trainer):
        super(TqdmCallback, self).on_train_end(trainer)
        self.clear_tasks()

    def on_before_backward(self, trainer, outputs):
        loss = trainer.extract_loss_from_outputs(outputs)
        loss = trainer.driver.tensor_to_numeric(loss, reduce='sum')
        self.loss += loss

        # 如果有额外的信息需要显示
        if self.extra_show_keys is not None:
            self.extra_show_keys.update(outputs)

    def on_train_batch_end(self, trainer):
        if trainer.global_forward_batches % self.print_every == 0:
            loss = self.loss / self.print_every
            self.loss = 0
            # 默认情况下进度条后只有 Loss 信息
            post_desc = f'Loss:{loss:.{self.loss_round_ndigit}f}'
            # 进度条后附加上用户希望的额外信息
            if self.extra_show_keys is not None:
                post_desc = post_desc + \
                    _get_beautiful_extra_string(
                        self.extra_show_keys.get_stat(), progress_type='tqdm')
            self.progress_bar.update(
                self.task2id['batch'], advance=self.print_every, refresh=True)
            self.progress_bar.set_postfix_str(self.task2id['batch'], post_desc)
            self.progress_bar.update(
                self.task2id['epoch'],
                advance=self.epoch_bar_update_advance,
                refresh=True)

    def on_evaluate_end(self, trainer, results):
        if len(results) == 0:
            return
        base_text = f'Eval. results on Epoch:{trainer.cur_epoch_idx}, ' \
                    f'Batch:{trainer.batch_idx_in_epoch}'
        text = ''
        if self.monitor is not None:
            if self.is_better_results(results, keep_if_better=True):
                self.record_better_monitor(trainer, results)
                if abs(self.monitor_value) != float('inf'):
                    text = '+' * self.num_signs + base_text + \
                           '+' * self.num_signs
        if len(text) == 0:
            text = '-' * self.num_signs + base_text + '-' * self.num_signs

        logger.info(text)
        results = {
            key: trainer.driver.tensor_to_numeric(value)
            for key, value in results.items() if not key.startswith('_')
        }
        if self.format_json:
            results = json.dumps(results, indent=2)
        logger.info(results)

    def clear_tasks(self):
        for key, taskid in self.task2id.items():
            self.progress_bar.destroy_task(taskid)
        self.task2id: Dict[str, Any] = {}
        self.loss = 0

    @property
    def name(self):  # progress bar的名称
        return 'tqdm'

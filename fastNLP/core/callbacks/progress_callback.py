import json
from typing import Union, Dict, Any, Sequence

__all__ = [
    'choose_progress_callback',
    'ProgressCallback',
    'RichCallback',
    'TqdmCallback',
    'RawTextCallback',
    'BaseExtraInfoModel'
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

    def record_better_monitor(self, trainer, results):
        self.best_monitor_step = trainer.global_forward_batches
        self.best_monitor_epoch = trainer.cur_epoch_idx
        self.best_results = self.itemize_results(results)

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


class BaseExtraInfoModel(object):
    r"""
    用来处理 ``extra_show_keys`` 中要显示的额外信息的类。这个类的作用是当 log 的输出周期大于 1 的时候可以对数据进行累积相加，到输出的时候再求平均。
    """
    @staticmethod
    def update(output_collection: Dict[str, Any], outputs: dict) -> None:
        r"""
        每个 batch 训练结束后调用，更新 output_collection 中的内容，使得 output_collection 中的内容是 outputs 中的内容的累积，默认为相加。

        :param output_collection: 累积了每次训练的结果。
        :param outputs: 一个 batch 的输出结果，已经根据提供的 ``extra_show_keys`` 进行了筛选，要将其累积到 ``output_collection`` 中。

        """
        for key in outputs.keys():
            if key in output_collection.keys():
                try:
                    output_collection[key] += outputs[key]
                except Exception as e:
                    logger.rank_zero_warning(f"Error found when accumulating key:{key}. \
                                             Variables of {type(output_collection[key])} cannot be added. \
                                             Please design your own `update` function.")
                    output_collection[key] = outputs[key]


    @staticmethod
    def get_stat(output_collection: Dict[str, Any], print_every: int) -> Dict:
        r"""
        每个 log 周期调用一次，将 output_collection 中的内容转换为需要显示的内容，默认为求平均。

        :param output_collection: 累积了每次训练的结果。
        :param print_every: log 周期，每隔多少个 batch 打印一次 log，可以用它来求平均。

        """
        if print_every > 1:
            for key in output_collection.keys():
                try:
                    output_collection[key] /= print_every
                except Exception as e:
                    logger.rank_zero_warning(f"Error found when computing the average of key:{key}. \
                                             Variables of {type(output_collection[key])} cannot be divided by an integer. \
                                             Please design your own `get_stat` function.")
        return output_collection


def _get_beautiful_extra_string(extra_info_collection: dict, progress_type: str) -> str:
    """
    用户有额外信息需要打印的情况下，将额外信息转换为字符串。
    """
    original = ''.join([f", {key}:{value}" for key, value in extra_info_collection.items()])
    if progress_type == 'rich':
        original = original.replace(',', ',\n')
    elif progress_type == 'tqdm':
        # TODO 使用 TQDM 进度条时，额外信息过长，控制台太小时会导致额外信息显示不全，且无法和 RICH 一样通过换行规避。
        pass
    return original

class RichCallback(ProgressCallback):
    """
    在训练过程中打印 *rich* progress bar 的 callback 。在 Trainer 中，默认就会使用这个 callback 来显示进度。如果需要定制这个 Callback 的
    参数，请通过实例化本 Callback 并传入到 Trainer 中实现。在打印 evaluate 的结果时，不会打印名称以 "_" 开头的内容。

    :param print_every: 多少个 batch 更新一次显示。
    :param loss_round_ndigit: 显示的 loss 保留多少位有效数字
    :param monitor: 监控的 metric 值。当检测到这个key的结果更好时，会打印出不同的颜色进行提示。

        * 为 ``None``
         将尝试使用 :class:`~fastNLP.core.controllers.Trainer` 中设置 `monitor` 值（如果有设置）。
        * 为 ``str``
         尝试直接使用该名称从 ``evaluation`` 结果中寻找，如果在 ``evaluation`` 结果中没有找到完全一致的名称，将
         使用 最长公共字符串算法 从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor`` 。
        * 为 :class:`Callable`
         接受参数为 ``evaluation`` 的结果(字典类型)，返回一个 ``float`` 值作为 ``monitor`` 的结果，如果当前结果中没有相关
         的 ``monitor`` 值请返回 ``None`` 。

    :param larger_better: 是否是 monitor 的结果越大越好。
    :param format_json: 是否格式化 json 再打印
    :param extra_show_keys: 每个 batch 训练结束后需要额外显示的内容。

        * 为 ``str``
         从 train_step 的返回 的 dict 中寻找该名称的内容，如果找到则打印出来。
        * 为 ``Sequence[str]``
         与 ``str`` 类似，但是可以打印多个内容。

    :param: extra_info_model: 当打印周期不为 1 的情况，是否希望额外打印的字段在打印周期间累积。

        * 为 ``None``
         不累积，仅打印当前轮次的额外信息
        * 为静态类 :class:`~fastNLP.core.callback.BaseExtraInfoModel` 及其子类，必须实现 ``update`` 和 ``get_stat`` 方法。
         在未进行打印的轮次，将额外信息相加累积，并在输出前求平均。可以手动重写 ``update`` 和 ``get_stat`` 方法来实现自己的累积方式。

    """

    def __init__(self, print_every: int = 1, loss_round_ndigit: int = 6, monitor: str = None,
                 larger_better: bool = True,
                 format_json=True,
                 extra_show_keys: Union[str, Sequence[str], None] = None,
                 extra_info_model: Union[type, None] = BaseExtraInfoModel):
        super().__init__(monitor=monitor, larger_better=larger_better, must_have_monitor=False)
        self.print_every = print_every
        self.progress_bar = f_rich_progress
        self.task2id = {}
        self.loss = 0
        self.loss_round_ndigit = loss_round_ndigit
        self.format_json = format_json
        # 用于在进度条中显示额外信息
        self.extra_show_keys = extra_show_keys if isinstance(extra_show_keys, Sequence) or extra_show_keys is None \
            else [extra_show_keys]
        self.extra_info_model = extra_info_model
        if not hasattr(self.extra_info_model, 'update') or not hasattr(self.extra_info_model, 'get_stat'):
            logger.rank_zero_warning(f"Your manually defined ExtraInfoModel does not implement the method: \
                    {', '.join([str(m) for m in ['update', 'get_stat'] if not hasattr(self.extra_info_model, m)])}.")
            self.extra_info_model = BaseExtraInfoModel
        self.extra_info_collection = None

    def on_after_trainer_initialized(self, trainer, driver):
        if not self.progress_bar.disable:
            self.progress_bar.set_disable(flag=trainer.driver.get_local_rank() != 0)
        super(RichCallback, self).on_after_trainer_initialized(trainer, driver)

    def on_train_begin(self, trainer):
        self.task2id['epoch'] = self.progress_bar.add_task(description=f'Epoch:{trainer.cur_epoch_idx}',
                                                           total=trainer.n_epochs,
                                                           completed=trainer.global_forward_batches / (
                                                                   trainer.n_batches + 1e-6) *
                                                                     trainer.n_epochs)

    def on_train_epoch_begin(self, trainer):
        self.epoch_bar_update_advance = self.print_every / (trainer.num_batches_per_epoch + 1e-6)
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

        # 如果有额外的信息需要显示
        if self.extra_show_keys is not None:
            if self.extra_info_collection is None or self.extra_info_model is None:
                self.extra_info_collection = {key: value for key, value in outputs.items()
                                              if key in self.extra_show_keys}
            else:
                self.extra_info_model.update(self.extra_info_collection, {key: value for key, value in outputs.items()
                                                                          if key in self.extra_show_keys})

    def on_train_batch_end(self, trainer):
        if trainer.global_forward_batches % self.print_every == 0:
            loss = self.loss / self.print_every
            self.loss = 0
            # 默认情况下进度条后只有 Loss 信息
            post_desc = f'Loss:{loss:.{self.loss_round_ndigit}f}'
            # 进度条后附加上用户希望的额外信息
            if self.extra_show_keys is not None:
                post_desc = post_desc + _get_beautiful_extra_string(self.extra_info_model.get_stat(
                    self.extra_info_collection, self.print_every), progress_type='rich')
                self.extra_info_collection = None
            self.progress_bar.update(self.task2id['batch'], description=f'Batch:{trainer.batch_idx_in_epoch}',
                                     advance=self.print_every,
                                     post_desc=post_desc, refresh=True)
            self.progress_bar.update(self.task2id['epoch'], description=f'Epoch:{trainer.cur_epoch_idx}',
                                     advance=self.epoch_bar_update_advance, refresh=True)

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
        self.progress_bar.console.rule(text_style + f"Eval. results on Epoch:{trainer.cur_epoch_idx}, "
                                                    f"Batch:{trainer.batch_idx_in_epoch}",
                                       style=rule_style, characters=characters)
        results = {key: trainer.driver.tensor_to_numeric(value) for key, value in results.items() if
                   not key.startswith('_')}
        if self.format_json:
            results = json.dumps(results)
            self.progress_bar.console.print_json(results)
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
    """
    通过向命令行打印进度的方式显示。在打印 evaluate 的结果时，不会打印名称以 "_" 开头的内容。

    :param print_every: 多少个 batch 更新一次显示。
    :param loss_round_ndigit: 显示的 loss 保留多少位有效数字
    :param monitor: 监控的 metric 值。当检测到这个key的结果更好时，会打印出不同的颜色进行提示。

        * 为 ``None``
            将尝试使用 :class:`~fastNLP.core.controllers.Trainer` 中设置 `monitor` 值（如果有设置）。
        * 为 ``str``
            尝试直接使用该名称从 ``evaluation`` 结果中寻找，如果在 ``evaluation`` 结果中没有找到完全一致的名称，将
            使用 最长公共字符串算法 从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor`` 。
        * 为 :class:`Callable`
            接受参数为 ``evaluation`` 的结果(字典类型)，返回一个 ``float`` 值作为 ``monitor`` 的结果，如果当前结果中没有相关
            的 ``monitor`` 值请返回 ``None`` 。
    :param larger_better: 是否是monitor的结果越大越好。
    :param format_json: 是否format json再打印。
    :param extra_show_keys: 每个 batch 训练结束后需要额外显示的内容。

        * 为 ``str``
         从 train_step 的返回 的 dict 中寻找该名称的内容，如果找到则打印出来。
        * 为 ``Sequence[str]``
         与 ``str`` 类似，但是可以打印多个内容。

    :param: extra_info_model: 当打印周期不为 1 的情况，是否希望额外打印的字段在打印周期间累积。

        * 为 ``None``
         不累积，仅打印当前轮次的额外信息
        * 为静态类 :class:`~fastNLP.core.callback.BaseExtraInfoModel` 及其子类，必须实现 ``update`` 和 ``get_stat`` 方法。
         在未进行打印的轮次，将额外信息相加累积，并在输出前求平均。可以手动重写 ``update`` 和 ``get_stat`` 方法来实现自己的累积方式。

    """

    def __init__(self, print_every: int = 1, loss_round_ndigit: int = 6, monitor: str = None,
                 larger_better: bool = True,
                 format_json=True,
                 extra_show_keys: Union[str, Sequence[str], None] = None,
                 extra_info_model: Union[type, None] = BaseExtraInfoModel):
        super().__init__(monitor=monitor, larger_better=larger_better, must_have_monitor=False)
        self.print_every = print_every
        self.task2id = {}
        self.loss = 0
        self.loss_round_ndigit = loss_round_ndigit
        self.set_monitor(monitor, larger_better)
        self.format_json = format_json
        self.num_signs = 10
        # 用于在进度条中显示额外信息
        self.extra_show_keys = extra_show_keys if isinstance(extra_show_keys, Sequence) or extra_show_keys is None \
            else [extra_show_keys]
        self.extra_info_model = extra_info_model
        if not hasattr(self.extra_info_model, 'update') or not hasattr(self.extra_info_model, 'get_stat'):
            logger.rank_zero_warning(f"Your manually defined ExtraInfoModel does not implement the method: \
                    {', '.join([str(m) for m in ['update', 'get_stat'] if not hasattr(self.extra_info_model, m)])}.")
            self.extra_info_model = BaseExtraInfoModel
        self.extra_info_collection = None

    def on_train_epoch_begin(self, trainer):
        logger.info('\n' + "*" * self.num_signs + f'Epoch:{trainer.cur_epoch_idx} starts' + '*' * self.num_signs)

    def on_before_backward(self, trainer, outputs):
        loss = trainer.extract_loss_from_outputs(outputs)
        loss = trainer.driver.tensor_to_numeric(loss, reduce='sum')
        self.loss += loss

        # 如果有额外的信息需要显示
        if self.extra_show_keys is not None:
            if self.extra_info_collection is None or self.extra_info_model is None:
                self.extra_info_collection = {key: value for key, value in outputs.items()
                                              if key in self.extra_show_keys}
            else:
                self.extra_info_model.update(self.extra_info_collection, {key: value for key, value in outputs.items()
                                                                          if key in self.extra_show_keys})

    def on_train_batch_end(self, trainer):
        if trainer.global_forward_batches % self.print_every == 0:
            loss = self.loss / self.print_every
            self.loss = 0
            # 默认情况下进度条后只有 Loss 信息
            post_desc = f'Loss:{loss:.{self.loss_round_ndigit}f}'
            # 进度条后附加上用户希望的额外信息
            if self.extra_show_keys is not None:
                post_desc = post_desc + _get_beautiful_extra_string(self.extra_info_model.get_stat(
                    self.extra_info_collection, self.print_every), progress_type='raw')
                self.extra_info_collection = None
            text = f'Epoch:{trainer.cur_epoch_idx}/{trainer.n_epochs}, Batch:{trainer.batch_idx_in_epoch}, ' \
                   + post_desc + \
                   f', finished {round(trainer.global_forward_batches / trainer.n_batches * 100, 2)}%.'
            logger.info(text)

    def on_evaluate_end(self, trainer, results):
        if len(results) == 0:
            return
        base_text = f'Eval. results on Epoch:{trainer.cur_epoch_idx}, Batch:{trainer.batch_idx_in_epoch}'
        text = ''
        if self.monitor is not None:
            if self.is_better_results(results, keep_if_better=True):
                self.record_better_monitor(trainer, results)
                if abs(self.monitor_value) != float('inf'):
                    text = '+' * self.num_signs + base_text + '+' * self.num_signs
        if len(text) == 0:
            text = '-' * self.num_signs + base_text + '-' * self.num_signs

        logger.info(text)
        results = {key: trainer.driver.tensor_to_numeric(value) for key, value in results.items() if
                   not key.startswith('_')}
        if self.format_json:
            results = json.dumps(results)
        logger.info(results)

    @property
    def name(self):  # progress bar的名称
        return 'raw'


class TqdmCallback(ProgressCallback):
    """
    在训练过程中打印 *tqdm* progress bar 的 callback 。在 Trainer 中，如果设置了 ``progress_bar='tqdm'`` 就会使用
    这个 callback 来显示进度。如果需要定制这个 Callback 的参数，请通过实例化本 Callback 并传入到 Trainer 中实现。在
    打印 evaluate 的结果时，不会打印名称以 "_" 开头的内容。

    :param print_every: 多少个 batch 更新一次显示。
    :param loss_round_ndigit: 显示的 loss 保留多少位有效数字
    :param monitor: 监控的 metric 值。当检测到这个key的结果更好时，会打印出不同的颜色进行提示。

        * 为 ``None``
         将尝试使用 :class:`~fastNLP.core.controllers.Trainer` 中设置 `monitor` 值（如果有设置）。
        * 为 ``str``
         尝试直接使用该名称从 ``evaluation`` 结果中寻找，如果在 ``evaluation`` 结果中没有找到完全一致的名称，将
         使用 最长公共字符串算法 从 ``evaluation`` 结果中找到最匹配的那个作为 ``monitor`` 。
        * 为 :class:`Callable`
         接受参数为 ``evaluation`` 的结果(字典类型)，返回一个 ``float`` 值作为 ``monitor`` 的结果，如果当前结果中没有相关
         的 ``monitor`` 值请返回 ``None`` 。
    :param larger_better: 是否是 monitor 的结果越大越好。
    :param format_json: 是否格式化 json 再打印
    :param extra_show_keys: 每个 batch 训练结束后需要额外显示的内容。

        * 为 ``str``
         从 train_step 的返回 的 dict 中寻找该名称的内容，如果找到则打印出来。
        * 为 ``Sequence[str]``
         与 ``str`` 类似，但是可以打印多个内容。

    :param: extra_info_model: 当打印周期不为 1 的情况，是否希望额外打印的字段在打印周期间累积。

        * 为 ``None``
         不累积，仅打印当前轮次的额外信息
        * 为静态类 :class:`~fastNLP.core.callback.BaseExtraInfoModel` 及其子类，必须实现 ``update`` 和 ``get_stat`` 方法。
         在未进行打印的轮次，将额外信息相加累积，并在输出前求平均。可以手动重写 ``update`` 和 ``get_stat`` 方法来实现自己的累积方式。
    """

    def __init__(self, print_every: int = 1, loss_round_ndigit: int = 6, monitor: str = None,
                 larger_better: bool = True,
                 format_json=True,
                 extra_show_keys: Union[str, Sequence[str], None] = None,
                 extra_info_model: Union[type, None] = BaseExtraInfoModel):
        super().__init__(monitor=monitor, larger_better=larger_better, must_have_monitor=False)
        self.print_every = print_every
        self.progress_bar = f_tqdm_progress
        self.task2id = {}
        self.loss = 0
        self.loss_round_ndigit = loss_round_ndigit
        self.format_json = format_json
        self.num_signs = 10
        # 用于在进度条中显示额外信息
        self.extra_show_keys = extra_show_keys if isinstance(extra_show_keys, Sequence) or extra_show_keys is None \
            else [extra_show_keys]
        self.extra_info_model = extra_info_model
        if not hasattr(self.extra_info_model, 'update') or not hasattr(self.extra_info_model, 'get_stat'):
            logger.rank_zero_warning(f"Your manually defined ExtraInfoModel does not implement the method: \
                    {', '.join([str(m) for m in ['update', 'get_stat'] if not hasattr(self.extra_info_model, m)])}.")
            self.extra_info_model = BaseExtraInfoModel
        self.extra_info_collection = None

    def on_train_begin(self, trainer):
        self.task2id['epoch'] = self.progress_bar.add_task(description=f'Epoch:{trainer.cur_epoch_idx}',
                                                           total=trainer.n_epochs, dynamic_ncols=True,
                                                           bar_format='{desc}: {percentage:3.0f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}, {postfix}]',
                                                           initial=trainer.global_forward_batches / (
                                                                   trainer.n_batches + 1e-6) * trainer.n_epochs)

    def on_train_epoch_begin(self, trainer):
        self.epoch_bar_update_advance = self.print_every / (trainer.num_batches_per_epoch + 1e-6)
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

        # 如果有额外的信息需要显示
        if self.extra_show_keys is not None:
            if self.extra_info_collection is None or self.extra_info_model is None:
                self.extra_info_collection = {key: value for key, value in outputs.items()
                                              if key in self.extra_show_keys}
            else:
                self.extra_info_model.update(self.extra_info_collection, {key: value for key, value in outputs.items()
                                                                          if key in self.extra_show_keys})

    def on_train_batch_end(self, trainer):
        if trainer.global_forward_batches % self.print_every == 0:
            loss = self.loss / self.print_every
            self.loss = 0
            # 默认情况下进度条后只有 Loss 信息
            post_desc = f'Loss:{loss:.{self.loss_round_ndigit}f}'
            # 进度条后附加上用户希望的额外信息
            if self.extra_show_keys is not None:
                post_desc = post_desc + _get_beautiful_extra_string(self.extra_info_model.get_stat(
                    self.extra_info_collection, self.print_every), progress_type='tqdm')
                self.extra_info_collection = None
            self.progress_bar.update(self.task2id['batch'], advance=self.print_every, refresh=True)
            self.progress_bar.set_postfix_str(self.task2id['batch'], post_desc)
            self.progress_bar.update(self.task2id['epoch'], advance=self.epoch_bar_update_advance, refresh=True)

    def on_evaluate_end(self, trainer, results):
        if len(results) == 0:
            return
        base_text = f'Eval. results on Epoch:{trainer.cur_epoch_idx}, Batch:{trainer.batch_idx_in_epoch}'
        text = ''
        if self.monitor is not None:
            if self.is_better_results(results, keep_if_better=True):
                self.record_better_monitor(trainer, results)
                if abs(self.monitor_value) != float('inf'):
                    text = '+' * self.num_signs + base_text + '+' * self.num_signs
        if len(text) == 0:
            text = '-' * self.num_signs + base_text + '-' * self.num_signs

        logger.info(text)
        results = {key: trainer.driver.tensor_to_numeric(value) for key, value in results.items() if
                   not key.startswith('_')}
        if self.format_json:
            results = json.dumps(results)
        logger.info(results)

    def clear_tasks(self):
        for key, taskid in self.task2id.items():
            self.progress_bar.destroy_task(taskid)
        self.task2id = {}
        self.loss = 0

    @property
    def name(self):  # progress bar的名称
        return 'tqdm'



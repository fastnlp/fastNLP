from typing import Union, List, Optional, Dict, Callable
from functools import partial
from dataclasses import is_dataclass
import sys

__all__ = [
    'Evaluator'
]

from fastNLP.core.drivers import Driver
from ..drivers.choose_driver import choose_driver
from .loops import Loop, EvaluateBatchLoop
from fastNLP.core.utils import auto_param_call, dataclass_to_dict, \
    match_and_substitute_params, f_rich_progress, flat_nest_dict
from fastNLP.core.metrics import Metric
from fastNLP.core.metrics.utils import _is_torchmetrics_metric, _is_paddle_metric, _is_allennlp_metric
from fastNLP.core.controllers.utils.utils import _TruncatedDataLoader
from fastNLP.core.utils.utils import _check_valid_parameters_number
from fastNLP.core.log import logger


class Evaluator:
    driver: Driver
    _evaluate_batch_loop: Loop

    def __init__(self, model, dataloaders, metrics: Optional[Union[Dict, Metric]] = None,
                 driver: Union[str, Driver] = 'torch', device: Optional[Union[int, List[int], str]] = None,
                 evaluate_batch_step_fn: Optional[callable] = None, evaluate_fn: Optional[str] = None,
                 input_mapping: Optional[Union[Callable, Dict]] = None,
                 output_mapping: Optional[Union[Callable, Dict]] = None, model_wo_auto_param_call: bool = False,
                 fp16: bool = False, verbose: int = 1, **kwargs):
        """
        用于对数据进行评测。

        :param model: 待测试的模型，如果传入的 driver 为 Driver 实例，该参数将被忽略。
        :param dataloaders: 待评测的数据集。如果为多个，请使用 dict 传入。
        :param metrics: 使用的 metric 。必须为 dict 类型，其中 key 为 metric 的名称，value 为一个 Metric 对象。支持 fastNLP 的
            metric ，torchmetrics，allennlpmetrics 等。
        :param driver: 使用 driver 。
        :param device: 使用的设备。
        :param evaluate_batch_step_fn: 定制每次 evaluate batch 执行的函数。该函数应接受的两个参数为 `evaluator` 和 `batch`，
            不需要有返回值；可以参考 fastNLP.core.controllers.loops.evaluate_batch_loop.EvaluateBatchLoop中的batch_step_fn函数。
        :param evaluate_fn: 用来控制 `Evaluator` 在评测的前向传播过程中是调用哪一个函数，例如是 `model.evaluate_step` 还是
            `model.forward`；(1) 如果该值是 None，那么我们会默认使用 `evaluate_step` 当做前向传播的函数，如果在模型中没有
            找到该方法，则使用 `model.forward` 函数；(2) 如果为 str 类型，则尝试从 model 中寻找该方法，找不到则报错。
        :param input_mapping: 对 dataloader 中输出的内容将通过 input_mapping 处理之后再输入到 model 以及 metric 中。如果针对
            model 和 metric 需要不同的 mapping，请考虑使用 evaluate_batch_step_fn 参数定制。
        :param output_mapping: 对 model 输出的内容，将通过 output_mapping 处理之后再输入到 metric 中。
        :param model_wo_auto_param_call: 是否关闭在训练时调用我们的 auto_param_call 来自动匹配 batch 和 forward 函数的参数的行为；
         如果该值为 True，并且当 batch 为字典时，我们会根据 forward 所需要的参数从 batch 中提取对应的对象，传入到 forward 函数中；如果该值
         为 False，那么我们会将 batch 直接透传给 forward 函数。注意上述逻辑同样应用于 `train_step`, `evaluate_step` 和 `test_step`；
        :param fp16: 是否使用 fp16 。
        :param verbose: 是否打印 evaluate 的结果。
        :param \**kwargs:
            See below
        :kwargs:
        * *model_use_eval_mode* (``bool``) -- 
            是否在 evaluate 的时候将 model 的状态设置成 eval 状态。在 eval 状态下，model 的
            dropout 与 batch normalization 将会关闭。默认为True。如果为 False，fastNLP 不会对 model 的 evaluate 状态做任何设置。无论
            该值是什么，fastNLP 都会在 evaluate 接受后将 model 的状态设置为 train 。
            TODO 还没完成。
        * *auto_tensor_conversion_for_metric* (``Union[bool]``) -- 
            是否自动将输出中的 tensor 适配到 metrics 支持的。例如 model 输出是
            paddlepaddle 的 tensor ，但是想利用 torchmetrics 的metric对象，当 auto_tensor_conversion_for_metric 为True时，fastNLP 将
            自动将输出中 paddle 的 tensor （其它非 tensor 的参数不做任何处理）转换为 pytorch 的 tensor 再输入到 metrics 中进行评测。 model 的
            输出 tensor 类型通过 driver 来决定，metrics 支持的输入类型由 metrics 决定。如果需要更复杂的转换，
            请使用 input_mapping、output_mapping 参数进行。
        * *use_dist_sampler* -- 
            是否使用分布式evaluate的方式。仅当 driver 为分布式类型时，该参数才有效。默认为根据 driver 是否支持
            分布式进行设置。如果为True，将使得每个进程上的 dataloader 自动使用不同数据，所有进程的数据并集是整个数据集。
        * *output_from_new_proc* -- 
            应当为一个字符串，表示在多进程的 driver 中其它进程的输出流应当被做如何处理；其值应当为以下之一：
            ["all", "ignore", "only_error"]；当该参数的值不是以上值时，该值应当表示一个文件夹的名字，我们会将其他 rank 的输出流重定向到
            log 文件中，然后将 log 文件保存在通过该参数值设定的文件夹中；默认为 "only_error"；
        * *progress_bar* -- 
            evaluate 的时候显示的 progress bar 。目前支持三种 [None, 'raw', 'rich', 'auto'], auto 表示如果检测
            到当前terminal为交互型则使用 rich，否则使用 raw。
        """

        self.model = model
        self.metrics = metrics
        self.driver = choose_driver(model, driver, device, fp16=fp16, model_wo_auto_param_call=model_wo_auto_param_call,
                                    **kwargs)

        if dataloaders is None:
            raise ValueError("Parameter `dataloaders` can not be None.")
        self.dataloaders = dataloaders
        self.device = device
        self.verbose = verbose

        if evaluate_batch_step_fn is not None:
            _check_valid_parameters_number(evaluate_batch_step_fn, ['evaluator', 'batch'], fn_name='evaluate_batch_step_fn')
        self.evaluate_batch_step_fn = evaluate_batch_step_fn

        self.input_mapping = input_mapping
        self.output_mapping = output_mapping

        if not isinstance(dataloaders, dict):
            dataloaders = {None: dataloaders}

        self.evaluate_batch_loop = EvaluateBatchLoop(batch_step_fn=evaluate_batch_step_fn)

        self.driver.setup()
        self.driver.barrier()

        self.separator = kwargs.get('separator', '#')
        self.model_use_eval_mode = kwargs.get('model_use_eval_mode', True)
        use_dist_sampler = kwargs.get("use_dist_sampler", self.driver.is_distributed())
        if use_dist_sampler:
            self._dist_sampler = "unrepeatdist"
        else:
            self._dist_sampler = None
        self._metric_wrapper = None
        _ = self.metrics_wrapper  # 触发检查

        if evaluate_fn is not None and not isinstance(evaluate_fn, str):
            raise TypeError("Parameter `evaluate_fn` can only be `str` type when it is not None.")
        self._evaluate_step, self._evaluate_step_signature_fn = \
            self.driver.get_model_call_fn("evaluate_step" if evaluate_fn is None else evaluate_fn)
        self.evaluate_fn = evaluate_fn

        self.dataloaders = {}
        for name, dl in dataloaders.items():  # 替换为正确的 sampler
            dl = self.driver.set_dist_repro_dataloader(dataloader=dl, dist=self._dist_sampler, reproducible=False)
            self.dataloaders[name] = dl

        self.progress_bar = kwargs.get('progress_bar', 'auto')
        if self.progress_bar == 'auto':
            self.progress_bar = 'raw' if f_rich_progress.dummy_rich else 'rich'

        self.driver.barrier()

    def run(self, num_eval_batch_per_dl: int = -1, **kwargs) -> Dict:
        """
        返回一个字典类型的数据，其中key为metric的名字，value为对应metric的结果。
            如果存在多个metric，一个dataloader的情况，key的命名规则是
                metric_indicator_name#metric_name
            如果存在多个数据集，一个metric的情况，key的命名规则是
                metric_indicator_name#metric_name#dataloader_name (其中 # 是默认的 separator ，可以通过 Evaluator 初始化参数修改)。
            如果存在多个metric，多个dataloader的情况，key的命名规则是
                metric_indicator_name#metric_name#dataloader_name
            其中 metric_indicator_name 可能不存在。

        :param num_eval_batch_per_dl: 每个 dataloader 测试多少个 batch 的数据，-1 为测试所有数据。
        :return:
        """
        assert isinstance(num_eval_batch_per_dl, int), "num_eval_batch_per_dl must be of int type."
        assert num_eval_batch_per_dl > 0 or num_eval_batch_per_dl == -1, "num_eval_batch_per_dl must be -1 or larger than 0."

        metric_results = {}
        self.reset()
        evaluate_context = self.driver.get_evaluate_context()
        self.driver.set_model_mode(mode='eval' if self.model_use_eval_mode else 'train')
        with evaluate_context():
            try:
                for dataloader_name, dataloader in self.dataloaders.items():
                    self.driver.barrier()
                    if num_eval_batch_per_dl != -1:
                        dataloader = _TruncatedDataLoader(dataloader, num_eval_batch_per_dl)
                    self.driver.set_sampler_epoch(dataloader, -1)
                    self.start_progress_bar(total=len(dataloader), dataloader_name=dataloader_name)
                    self.cur_dataloader_name = dataloader_name
                    results = self.evaluate_batch_loop.run(self, dataloader)
                    self.remove_progress_bar(dataloader_name)
                    metric_results[dataloader_name] = results
                    self.reset()
                    self.driver.barrier()
            except BaseException as e:
                raise e
            finally:
                self.finally_progress_bar()
        if len(metric_results) > 0:  # 如果 metric 不为 None 需要 print 。
            metric_results = flat_nest_dict(metric_results, separator=self.separator, compress_none_key=True, top_down=False)
            if self.verbose:
                if self.progress_bar == 'rich':
                    f_rich_progress.print(metric_results)
                else:
                    logger.info(metric_results)
        self.driver.set_model_mode(mode='train')

        return metric_results

    def start_progress_bar(self, total: int, dataloader_name):
        if self.progress_bar == 'rich':
            if dataloader_name is None:
                desc = f'Eval. Batch:0'
            else:
                desc = f'Eval. on {dataloader_name} Batch:0'
            self._rich_task_id = f_rich_progress.add_task(description=desc, total=total)
        elif self.progress_bar == 'raw':
            desc = 'Evaluation starts'
            if dataloader_name is not None:
                desc += f' on {dataloader_name}'
            logger.info('\n' + "*" * 10 + desc + '*' * 10)

    def update_progress_bar(self, batch_idx, dataloader_name, **kwargs):
        if dataloader_name is None:
            desc = f'Eval. Batch:{batch_idx}'
        else:
            desc = f'Eval. on {dataloader_name} Batch:{batch_idx}'
        if self.progress_bar == 'rich':
            assert hasattr(self, '_rich_task_id'), "You must first call `start_progress_bar()` before calling " \
                                                   "update_progress_bar()"
            f_rich_progress.update(self._rich_task_id, description=desc, post_desc=kwargs.get('post_desc', ''),
                                   advance=kwargs.get('advance', 1), refresh=kwargs.get('refresh', True),
                                   visible=kwargs.get('visible', True))
        elif self.progress_bar == 'raw':
            if self.verbose > 1:
                logger.info(desc)

    def remove_progress_bar(self, dataloader_name):
        if self.progress_bar == 'rich' and hasattr(self, '_rich_task_id'):
            f_rich_progress.destroy_task(self._rich_task_id)
            delattr(self, '_rich_task_id')
        elif self.progress_bar == 'raw':
            desc = 'Evaluation ends'
            if dataloader_name is not None:
                desc += f' on {dataloader_name}'
            logger.info("*" * 10 + desc + '*' * 10 + '\n')

    def finally_progress_bar(self):
        if self.progress_bar == 'rich' and hasattr(self, '_rich_task_id'):
            f_rich_progress.destroy_task(self._rich_task_id)
            delattr(self, '_rich_task_id')

    @property
    def evaluate_batch_loop(self):
        return self._evaluate_batch_loop

    @evaluate_batch_loop.setter
    def evaluate_batch_loop(self, loop: Loop):
        if self.evaluate_batch_step_fn is not None:
            logger.rank_zero_warning("`evaluate_batch_step_fn` was customized in the Evaluator initialization, it will be ignored "
                           "when the `evaluate_batch_loop` is also customized.")
        self._evaluate_batch_loop = loop

    def reset(self):
        """
        调用所有 metric 的 reset() 方法，清除累积的状态。

        Returns:

        """
        self.metrics_wrapper.reset()

    def update(self, batch, outputs):
        """
        自动调用所有 metric 的 update 方法，会根据不同 metric 的参数列表进行匹配传参。

        :param batch: 一般是来自于 DataLoader 的输出，如果不为 dict 类型的话，该值将被忽略。
        :param outputs: 一般是来自于模型的输出。类别应为 dict 或者 dataclass 类型。
        :return:
        """
        self.metrics_wrapper.update(batch, outputs)

    def get_metric(self) -> Dict:
        """
        调用所有 metric 的 get_metric 方法，并返回结果。其中 key 为 metric 的名称，value 是各个 metric 的结果。

        :return:
        """
        return self.metrics_wrapper.get_metric()

    @property
    def metrics_wrapper(self):
        """
        由于需要保持 Evaluator 中 metrics 对象与用户传入的 metrics 保持完全一致（方便他在 evaluate_batch_step_fn ）中使用，同时也为了支持
        不同形式的 metric（ fastNLP 的 metric/torchmetrics 等），所以 Evaluator 在进行 metric 操作的时候都调用 metrics_wrapper
        进行操作。

        Returns:
        """
        if self._metric_wrapper is None:
            self._metric_wrapper = _MetricsWrapper(self.metrics, evaluator=self)
        return self._metric_wrapper

    def evaluate_step(self, batch):
        """
        将 batch 传递到model中进行处理，根据当前 evaluate_fn 选择进行 evaluate 。会将返回结果经过 output_mapping 处理后再
            返回。

        :param batch: {evaluate_fn} 函数支持的输入类型
        :return: {evaluate_fn} 函数的输出结果，如果有设置 output_mapping ，将是 output_mapping 之后的结果。
        """
        outputs = self.driver.model_call(batch, self._evaluate_step, self._evaluate_step_signature_fn)
        outputs = match_and_substitute_params(self.output_mapping, outputs)
        return outputs

    @property
    def metrics(self):
        """
        返回用户传入的 metrics 对象。

        :return:
        """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        self._metrics = metrics

    def move_data_to_device(self, batch):
        return self.driver.move_data_to_device(batch)


class _MetricsWrapper:
    """
    注意 metrics 的输入只支持：Dict[str, Metric]；
    并且通过对 update() , reset() , get_metric() 函数的封装，实现支持 fastNLP 的 metric 以及 torchmetrics 或者更多。

    """

    def __init__(self, metrics, evaluator):
        self.evaluator = evaluator
        self._metrics = []
        self._metric_names = []
        if metrics is not None:
            if not isinstance(metrics, Dict):
                raise TypeError("Parameter `metrics` can only be `Dict` type.")
            for metric_name, metric in metrics.items():
                # 因为 torchmetrics 是一个 nn.Module，因此我们需要先将其移到对应的机器上；
                if _is_torchmetrics_metric(metric):
                    # torchmetrics 是默认自动开启了多卡的
                    evaluator.driver.move_model_to_device(metric, evaluator.driver.data_device)
                elif isinstance(metric, Metric):
                    # 如果数据是分布式的，但是不aggregate的话可能有问题
                    if evaluator._dist_sampler is not None and metric.aggregate_when_get_metric is False:
                        logger.warning_once(
                            "You have replace the sampler as distributed sampler when evaluation, but your "
                            f"metric {metric_name}:{metric.__class__.__name__}' `aggregate_when_get_metric` is False.")
                    if metric.aggregate_when_get_metric is None:
                        metric.aggregate_when_get_metric = evaluator._dist_sampler is not None

                    metric.to(evaluator.driver.data_device)
                self._metric_names.append(metric_name)
                self._metrics.append(metric)

    def update(self, batch, outputs):
        if is_dataclass(outputs):
            outputs = dataclass_to_dict(outputs)
        for metric in self._metrics:
            args = []
            if not isinstance(batch, dict):
                logger.warning_once(
                    f"The output of the DataLoader is of type:`{type(batch)}`, fastNLP will only depend on "
                    f"the output of model to update metric.")
            else:
                args.append(batch)
            if not isinstance(outputs, dict):
                raise RuntimeError(f"The output of your model is of type:`{type(outputs)}`, please either directly"
                                   f" return a dict from your model or use `output_mapping` to convert it into dict type.")
            if isinstance(metric, Metric):
                # 这样在 auto_param_call 报错的时候才清晰。
                auto_param_call(metric.update, outputs, *args, signature_fn=metric.update.__wrapped__)
            elif _is_torchmetrics_metric(metric):
                auto_param_call(metric.update, outputs, *args, signature_fn=metric.update.__wrapped__)
            elif _is_allennlp_metric(metric):
                auto_param_call(metric.__call__, outputs, *args)
            elif _is_paddle_metric(metric):
                res = auto_param_call(metric.compute, outputs, *args)
                metric.update(res)

    def reset(self):
        for metric in self._metrics:
            if _is_allennlp_metric(metric):
                metric.get_metric(reset=True)
            elif _is_torchmetrics_metric(metric) or _is_paddle_metric(metric) or isinstance(metric, Metric):
                metric.reset()

    def get_metric(self) -> Dict:
        """
        调用各个 metric 得到 metric 的结果。并使用 {'metric_name1': metric_results, 'metric_name2': metric_results} 的形式
            返回。

        :return:
        """
        results = {}
        for metric_name, metric in zip(self._metric_names, self._metrics):
            if isinstance(metric, Metric):
                _results = metric.get_metric()
            elif _is_allennlp_metric(metric):
                _results = metric.get_metric(reset=False)
            elif _is_torchmetrics_metric(metric):
                _results = metric.compute()
            elif _is_paddle_metric(metric):
                _results = metric.accumulate()
            else:
                raise RuntimeError(f"Not support `{type(metric)}` for now.")
            results[metric_name] = _results
        return results

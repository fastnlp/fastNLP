from typing import Union, List, Optional, Dict, Callable
from functools import partial
from dataclasses import is_dataclass
import sys


__all__ = [
    'Evaluator'
]

from fastNLP.core.drivers import Driver
from fastNLP.core.drivers.utils import choose_driver
from .loops import Loop, EvaluateBatchLoop
from fastNLP.core.utils import check_fn_not_empty_params, auto_param_call, dataclass_to_dict, \
    match_and_substitute_params, f_rich_progress
from fastNLP.core.metrics import Metric
from fastNLP.core.metrics.utils import _is_torchmetrics_metric, _is_paddle_metric, _is_allennlp_metric
from fastNLP.core.controllers.utils.utils import _TruncatedDataLoader
from fastNLP.core.log import logger


class Evaluator:
    """
    1. 我们目前不直接提供每一个 metric 对应一个或者特殊的多个 dataloader 的功能，默认就是所有 metric 处理所有 dataloader，如果用户有这种
    需求，请使用多个 Tester 进行操作；
    2. Trainer 的 validate dataloader 只允许传进去一个，而 Tester 则可以多个；因为 Trainer 涉及到保存 topk 模型的逻辑，而 Tester
    则只需要给出评测的结果即可；

    """
    driver: Driver
    _evaluate_batch_loop: Loop

    def __init__(
            self,
            model,
            dataloaders,
            metrics: Optional[Union[Dict, Metric]] = None,
            driver: Union[str, Driver] = 'single',
            device: Optional[Union[int, List[int], str]] = None,
            batch_step_fn: Optional[callable] = None,
            mode: str = "validate",
            input_mapping: Optional[Union[Callable, Dict]] = None,
            output_mapping: Optional[Union[Callable, Dict]] = None,
            fp16: Optional[bool] = False,
            verbose: int = 1,
            **kwargs
    ):
        """

        :param dataloaders:
        :param model:
        :param metrics: 使用的 metric 。必须为 dict 类型，其中 key 为 metric 的名称，value 为一个 Metric 对象。支持 fastNLP 的
            metric ，torchmetrics，allennlpmetrics等。
        :param driver: 使用 driver 。
        :param device: 使用的设备。
        :param batch_step_fn: callable的对象，接受 (evaluator, batch) 作为参数，其中 evaluator 为 Evaluator 对象，batch 为
            DataLoader 中返回的对象。一个 batch_step_fn 的例子可参考 fastNLP.core.controller.loops.evaluate_batch_loop 的
            batch_step_fn 函数。
        :param mode: 可选 ["validate", "test"], 当为 "validate" 时将首先尝试寻找 model 是否有 validate_step 函数，没有的话则尝试
            寻找 test_step 函数，都没找到则使用 model 的前向运算函数。当为 "test" 是将首先尝试寻找 model 是否有 test_step 函数，
            没有的话尝试 "validate_step"  函数，都没找到则使用 model 的前向运算函数。
        :param input_mapping: 对 dataloader 中输出的内容将通过 input_mapping 处理之后再输入到 model 以及 metric 中
        :param output_mapping: 对 model 输出的内容，将通过 output_mapping 处理之后再输入到 metric 中。
        :param fp16: 是否使用 fp16 。
        :param verbose: 是否打印 evaluate 的结果。
        :param kwargs:
            bool model_use_eval_mode: 是否在 evaluate 的时候将 model 的状态设置成 eval 状态。在 eval 状态下，model 的dropout
             与 batch normalization 将会关闭。默认为True。
            Union[bool] auto_tensor_conversion_for_metric: 是否自动将输出中的
             tensor 适配到 metrics 支持的。例如 model 输出是 paddlepaddle 的 tensor ，但是想利用 torchmetrics 的metric对象，
             当 auto_tensor_conversion_for_metric 为True时，fastNLP 将自动将输出中 paddle 的 tensor （其它非 tensor 的参数
             不做任何处理）转换为 pytorch 的 tensor 再输入到 metrics 中进行评测。 model 的输出 tensor 类型通过 driver 来决定，
             metrics 支持的输入类型由 metrics 决定。如果需要更复杂的转换，请使用 input_mapping、output_mapping 参数进行。
            use_dist_sampler: 是否使用分布式evaluate的方式。仅当 driver 为分布式类型时，该参数才有效。如果为True，将使得每个进程上
             的 dataloader 自动使用不同数据，所有进程的数据并集是整个数据集。请确保使用的 metrics 支持自动分布式累积。
            output_from_new_proc: 应当为一个字符串，表示在多进程的 driver 中其它进程的输出流应当被做如何处理；其值应当为以下之一：
             ["all", "ignore", "only_error"]；当该参数的值不是以上值时，该值应当表示一个文件夹的名字，我们会将其他 rank 的输出流重定向到
             log 文件中，然后将 log 文件保存在通过该参数值设定的文件夹中；默认为 "only_error"；
            progress_bar: evaluate 的时候显示的 progress bar 。目前支持三种 [None, 'raw', 'rich', 'auto'], auto 表示如果检测
                到当前terminal为交互型则使用 rich，否则使用 raw。
        """

        self.model = model
        self.metrics = metrics

        self.driver = choose_driver(model, driver, device, fp16=fp16, **kwargs)

        self.device = device
        self.verbose = verbose

        assert check_fn_not_empty_params(batch_step_fn, 2), "Parameter `batch_step_fn` should be a callable object with " \
                                                             "two parameters."
        self.batch_step_fn = batch_step_fn

        self.mode = mode
        assert mode in {'validate', 'test'}, "Parameter `mode` should only be 'validate' or 'test'."

        self.input_mapping = input_mapping
        self.output_mapping = output_mapping

        if not isinstance(dataloaders, dict):
            dataloaders = {None: dataloaders}
        if mode == "validate":
            self._evaluate_step = self.driver.validate_step
            self.driver.set_dataloader(validate_dataloaders=dataloaders)
        else:
            self._evaluate_step = self.driver.test_step
            self.driver.set_dataloader(test_dataloaders=dataloaders)
        self.mode = mode
        self.evaluate_batch_loop = EvaluateBatchLoop(batch_step_fn=batch_step_fn)
        self.separator = kwargs.get('separator', '#')
        self.model_use_eval_mode = kwargs.get('model_use_eval_mode', True)
        use_dist_sampler = kwargs.get("use_dist_sampler", False)  # 如果是 Evaluator 自身的默认值的话，应当为 False；
        if use_dist_sampler:
            self._dist_sampler = "unrepeatdist"
        else:
            self._dist_sampler = None
        self._metric_wrapper = None
        _ = self.metrics_wrapper  # 触发检查

        assert self.driver.has_validate_dataloaders() or self.driver.has_test_dataloaders()
        self.driver.setup()
        self.driver.barrier()

        self.dataloaders = {}
        for name, dl in dataloaders.items():  # 替换为正确的 sampler
            dl = self.driver.set_dist_repro_dataloader(dataloader=dl, dist=self._dist_sampler, reproducible=False)
            self.dataloaders[name] = dl

        self.progress_bar = kwargs.get('progress_bar', 'auto')
        if self.progress_bar == 'auto':
            self.progress_bar = 'rich' if (sys.stdin and sys.stdin.isatty()) else 'raw'

        self.driver.barrier()

    def run(self, num_eval_batch_per_dl: int = -1) -> Dict:
        """
        返回一个字典类型的数据，其中key为metric的名字，value为对应metric的结果。
        如果存在多个metric，一个dataloader的情况，key的命名规则是
            metric_indicator_name#metric_name
        如果存在多个数据集，一个metric的情况，key的命名规则是
            metric_indicator_name#dataloader_name (其中 # 是默认的 separator ，可以通过 Evaluator 初始化参数修改)。
        如果存在多个metric，多个dataloader的情况，key的命名规则是
            metric_indicator_name#metric_name#dataloader_name
        :param num_eval_batch_per_dl: 每个 dataloader 测试多少个 batch 的数据，-1 为测试所有数据。

        :return:
        """
        assert isinstance(num_eval_batch_per_dl, int), "num_eval_batch_per_dl must be of int type."
        assert num_eval_batch_per_dl > 0 or num_eval_batch_per_dl == -1, "num_eval_batch_per_dl must be -1 or larger than 0."

        self.driver.check_evaluator_mode(self.mode)

        if self.mode == 'validate':
            assert self.driver.has_validate_dataloaders()
        else:
            assert self.driver.has_test_dataloaders()

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
                    metric_results.update(results)
                    self.reset()
                    self.driver.barrier()
            except BaseException as e:
                raise e
            finally:
                self.finally_progress_bar()
        self.driver.set_model_mode(mode='train')
        if self.verbose:
            if self.progress_bar == 'rich':
                f_rich_progress.print(metric_results)
            else:
                logger.info(metric_results)

        return metric_results

    def start_progress_bar(self, total:int, dataloader_name):
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
            if self.verbose>1:
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
    def eval_dataloaders(self):
        if self.mode == "validate":
            return self.driver.validate_dataloaders
        else:
            return self.driver.test_dataloaders

    @property
    def evaluate_batch_loop(self):
        return self._evaluate_batch_loop

    @evaluate_batch_loop.setter
    def evaluate_batch_loop(self, loop: Loop):
        if self.batch_step_fn is not None:
            logger.warning("`batch_step_fn` was customized in the Evaluator initialization, it will be ignored "
                           "when the `evaluate_batch_loop` is also customized.")
        self._evaluate_batch_loop = loop

    def reset(self):
        """
        调用所有 metric 的 reset() 方法，清除累积的状态。

        Returns:

        """
        self.metrics_wrapper.reset()

    def update(self, *args, **kwargs):
        """
        调用所有metric的 update 方法，对当前 batch 的结果进行累积，会根据相应 metric 的参数列表进行匹配传参。

        :param args:
        :param kwargs:
        :return:
        """
        self.metrics_wrapper.update(*args, **kwargs)

    def get_dataloader_metric(self, dataloader_name:Optional[str]='') -> Dict:
        """
        获取当前dataloader的metric结果

        :param str dataloader_name: 当前dataloader的名字
        :return:
        """
        return self.metrics_wrapper.get_metric(dataloader_name=dataloader_name, separator=self.separator)

    @property
    def metrics_wrapper(self):
        """
        由于需要保持 Evaluator 中 metrics 对象与用户传入的 metrics 保持完全一致（方便他在 batch_step_fn ）中使用，同时也为了支持
        不同形式的 metric（ fastNLP 的 metric/torchmetrics 等），所以 Evaluator 在进行 metric 操作的时候都调用 metrics_wrapper
        进行操作。

        Returns:
        """
        if self._metric_wrapper is None:
            self._metric_wrapper = _MetricsWrapper(self.metrics, evaluator=self)
        return self._metric_wrapper

    def evaluate_step(self, batch):
        """
        将 batch 传递到model中进行处理，根据当前 mode 选择进行 evaluate 还是 test 。会将返回结果经过 output_mapping 处理后再
            返回。

        :param batch:
        :return:
        """
        outputs = self._evaluate_step(batch)
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
                    if evaluator._dist_sampler is not None and evaluator.driver.is_distributed() \
                            and metric.aggregate_when_get_metric is False:
                        logger.warning("You have replace the sampler as distributed sampler when evaluation, but your "
                                       f"metric:{metric_name}' `aggregate_when_get_metric` is False.")
                    if evaluator._dist_sampler is None and evaluator.driver.is_distributed() \
                        and metric.aggregate_when_get_metric is True:
                        pass  # 这种情况无所谓，因为
                    metric.to(evaluator.driver.data_device)
                self._metric_names.append(metric_name)
                self._metrics.append(metric)

    def update(self, batch, outputs):
        if is_dataclass(outputs):
            outputs = dataclass_to_dict(outputs)
        for metric in self._metrics:
            if not isinstance(batch, dict):
                raise RuntimeError(f"When the output of the DataLoader is of type:`{type(batch)}`, please either directly"
                                   f" return a dict from your DataLoader or use `input_mapping` to convert it into dict type.")
            if not isinstance(outputs, dict):
                raise RuntimeError(f"When the output of your model is of type:`{type(batch)}`, please either directly"
                                   f" return a dict from your model or use `output_mapping` to convert it into dict type.")
            if isinstance(metric, Metric):
                auto_param_call(metric.update, batch, outputs)
            elif _is_torchmetrics_metric(metric):
                auto_param_call(metric.update, batch, outputs)
            elif _is_allennlp_metric(metric):
                auto_param_call(metric.__call__, batch, outputs)
            elif _is_paddle_metric(metric):
                res = auto_param_call(metric.compute, batch, outputs)
                metric.update(res)

    def reset(self):
        for metric in self._metrics:
            if _is_allennlp_metric(metric):
                metric.get_metric(reset=True)
            elif _is_torchmetrics_metric(metric) or _is_paddle_metric(metric) or isinstance(metric, Metric):
                metric.reset()

    def get_metric(self, dataloader_name:str, separator:str) -> Dict:
        """
        将所有 metric 结果展平到一个一级的字典中，这个字典中 key 的命名规则是
            indicator_name{separator}metric_name{separator}dataloader_name
        例如: f1#F1PreRec#dev

        :param dataloader_name: 当前metric对应的dataloader的名字。若为空，则不显示在最终的key上面。
        :param separator: 用于间隔不同称呼。
        :return: 返回一个一级结构的字典，其中 key 为区别一个 metric 的名字，value 为该 metric 的值；
        """
        results = {}
        for metric_name, metric in zip(self._metric_names, self._metrics):
            if isinstance(metric, Metric):
                _results = metric.get_metric()
            elif _is_allennlp_metric(metric):
                _results = metric.get_metric(reset=False)
            elif _is_torchmetrics_metric(metric):
                _results = metric.compute()
                # 我们规定了 evaluator 中的 metrics 的输入只能是一个 dict，这样如果 metric 是一个 torchmetrics 时，如果 evaluator
                #  没有传入 func_post_proc，那么我们就自动使用该 metric 的 metric name 当做其的 indicator name 将其自动转换成一个字典；
            elif _is_paddle_metric(metric):
                _results = metric.accumulate()
            if not isinstance(_results, Dict):
                name = _get_metric_res_name(dataloader_name, metric_name, '', separator)
                results[name] = _results
            else:
                for indicator_name, value in _results.items():
                    name = _get_metric_res_name(dataloader_name, metric_name, indicator_name, separator)
                    results[name] = value

        return results


def _get_metric_res_name(dataloader_name: Optional[str], metric_name: str, indicator_name: str, separator='#') -> str:
    """

    :param dataloader_name: dataloder的名字
    :param metric_name: metric的名字
    :param indicator_name: metric中的各项metric名称，例如f, precision, recall
    :param separator: 用以间隔不同对象的间隔符
    :return:
    """
    names = []
    if indicator_name:
        names.append(indicator_name)
    if metric_name:
        names.append(metric_name)
    if dataloader_name:
        names.append(dataloader_name)
    if len(names) == 0:
        raise RuntimeError("You cannot use empty `dataloader_name`, `metric_name`, and `monitor` simultaneously.")
    return separator.join(names)
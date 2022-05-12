from typing import Optional, Callable, Dict

__all__ = [
    'EvaluateBatchLoop'
]

from .loop import Loop
from fastNLP.core.log import logger
from fastNLP.core.utils import match_and_substitute_params


class EvaluateBatchLoop(Loop):
    r"""
    ``EvaluateBatchLoop`` 针对一个 dataloader 的数据完成一个 epoch 的评测迭代过程；

    :param batch_step_fn: 您可以传入该参数来替换默认的 bath_step_fn；
    """
    def __init__(self, batch_step_fn:Optional[Callable]=None):
        if batch_step_fn is not None:
            self.batch_step_fn = batch_step_fn

    def run(self, evaluator, dataloader) -> Dict:
        r"""
        需要返回在传入的 dataloader 中的 evaluation 结果

        :param evaluator: Evaluator 对象
        :param dataloader: 当前需要进行评测的dataloader
        :return:
        """
        iterator = iter(dataloader)
        batch_idx = 0
        while True:
            try:
                batch = next(iterator)
            except StopIteration:
                break
            try:
                batch = match_and_substitute_params(evaluator.input_mapping, batch)
                batch = evaluator.move_data_to_device(batch)

                self.batch_step_fn(evaluator, batch)
                batch_idx += 1
                evaluator.update_progress_bar(batch_idx, evaluator.cur_dataloader_name)

            except BaseException as e:
                if callable(getattr(dataloader, 'get_batch_indices', None)):
                    indices = dataloader.get_batch_indices()
                    logger.error(f"Exception happens when evaluating on samples: {indices}")
                raise e
        # 获取metric结果。返回的dict内容示例为{'metric_name1': metric_results, 'metric_name2': metric_results, ...}
        results = evaluator.get_metric()
        return results

    @staticmethod
    def batch_step_fn(evaluator, batch):
        r"""
        针对一个 batch 的数据的评测过程；

        :param evaluator: Evaluator 对象
        :param batch: 当前需要评测的一个 batch 的数据；
        """
        outputs = evaluator.evaluate_step(batch)  # 将batch输入到model中得到结果
        evaluator.update(batch, outputs)  # evaluator将根据metric的形参名字从batch/outputs中取出对应的值进行赋值

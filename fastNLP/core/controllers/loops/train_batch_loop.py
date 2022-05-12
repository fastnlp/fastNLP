__all__ = [
    'TrainBatchLoop'
]

from typing import Optional, Callable

from .loop import Loop
from fastNLP.core.log import logger
from fastNLP.core.utils import match_and_substitute_params
from fastNLP.core.utils.exceptions import EarlyStopException


class TrainBatchLoop(Loop):
    r"""
    ``TrainBatchLoop`` 针对一个 dataloader 的数据完成一个 epoch 的训练迭代过程；

    :param batch_step_fn: 您可以传入该参数来替换默认的 bath_step_fn；
    """

    def __init__(self, batch_step_fn: Optional[Callable] = None):
        if batch_step_fn is not None:
            self.batch_step_fn = batch_step_fn

    def run(self, trainer, dataloader):
        r"""
        对传入的 dataloader 进行一个 epoch 的主要的训练的循环过程；

        .. note::

            您不需要自己主动地调用该方法，``Trainer`` 会负责调用该方法来完成训练过程；

        :param trainer: ``Trainer`` 实例；
        :param dataloader: 当前训练所使用的 dataloader；
        """
        get_batch_indices = dataloader.get_batch_indices if callable(getattr(dataloader, 'get_batch_indices', None))\
            else lambda *args, **kwargs: None
        dataloader = iter(dataloader)
        while trainer.batch_idx_in_epoch<=trainer.num_batches_per_epoch:
            try:
                trainer.on_fetch_data_begin()
                batch = next(dataloader)
                indices = get_batch_indices()
            except StopIteration:
                break

            try:
                trainer.on_fetch_data_end()
                batch = match_and_substitute_params(trainer.input_mapping, batch)
                batch = trainer.move_data_to_device(batch)

                trainer.on_train_batch_begin(batch, indices)
                with trainer.get_no_sync_context():  # 在多卡的时候可能需要关闭 sync
                    self.batch_step_fn(trainer, batch)
                trainer.global_forward_batches += 1
                trainer.batch_idx_in_epoch += 1

                trainer.check_batch_step_fn()
                trainer.on_train_batch_end()
            except BaseException as e:
                if indices is not None and not isinstance(e, (EarlyStopException, KeyboardInterrupt)):
                    logger.error(f"Exception happens when running on samples: {indices}")
                raise e
            trainer.step_evaluate()
        trainer.batch_idx_in_epoch = 0

    @staticmethod
    def batch_step_fn(trainer, batch):
        r"""
        针对一个 batch 的数据的训练过程；

        :param trainer: ``Trainer`` 实例；
        :param batch: 一个 batch 的数据；
        """
        outputs = trainer.train_step(batch)
        trainer.backward(outputs)
        trainer.step()
        trainer.zero_grad()





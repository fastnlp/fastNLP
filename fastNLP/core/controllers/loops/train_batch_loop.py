__all__ = [
    'TrainBatchLoop'
]

from typing import Optional, Callable

from .loop import Loop
from fastNLP.core.log import logger
from fastNLP.core.utils import match_and_substitute_params


class TrainBatchLoop(Loop):
    def __init__(self, batch_step_fn: Optional[Callable] = None):
        if batch_step_fn is not None:
            self.batch_step_fn = batch_step_fn

    def run(self, trainer, dataloader):
        get_batch_indices = dataloader.get_batch_indices if callable(getattr(dataloader, 'get_batch_indices', None))\
            else lambda *args, **kwargs: None
        dataloader = iter(dataloader)
        indices = None
        while True:
            try:
                trainer.on_fetch_data_begin()
                batch = next(dataloader)
                batch = match_and_substitute_params(trainer.input_mapping, batch)
                indices = get_batch_indices()
                batch = trainer.move_data_to_device(batch)
                trainer.on_fetch_data_end()
            except StopIteration:
                break
            except BaseException as e:  # TODO 把这里的信息写入进去
                if indices:
                    logger.debug(f"The following exception happens when running on samples: {indices}")
                raise e

            trainer.on_train_batch_begin(batch, indices)
            self.batch_step_fn(trainer, batch)
            trainer.global_forward_batches += 1
            trainer.batch_idx_in_epoch += 1

            trainer.check_batch_step_fn()
            trainer.on_train_batch_end()
            trainer.step_validate()
        trainer.batch_idx_in_epoch = 0

    @staticmethod
    def batch_step_fn(trainer, batch):
        outputs = trainer.train_step(batch)
        trainer.backward(outputs)
        trainer.step()
        trainer.zero_grad()





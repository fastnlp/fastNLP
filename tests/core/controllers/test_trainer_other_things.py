import pytest

from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.callbacks import Events
from tests.helpers.utils import magic_argv_env_context


@magic_argv_env_context
def test_trainer_torch_without_evaluator():
    @Trainer.on(Events.on_train_epoch_begin(every=10), marker="test_trainer_other_things")
    def fn1(trainer):
        pass

    @Trainer.on(Events.on_train_batch_begin(every=10), marker="test_trainer_other_things")
    def fn2(trainer, batch, indices):
        pass

    with pytest.raises(BaseException):
        @Trainer.on(Events.on_train_batch_begin(every=10), marker="test_trainer_other_things")
        def fn3(trainer, batch):
            pass





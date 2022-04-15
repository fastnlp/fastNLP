import pytest
import numpy as np

from fastNLP.core.callbacks import TorchWarmupCallback, Callback
from fastNLP import Trainer

from tests.helpers.callbacks.prepare_trainer_args_for_torch_test import get_trainer_args


class RecordLrCallback(Callback):
    def __init__(self):
        self.lrs = []

    def on_after_optimizers_step(self, trainer, optimizers):
        self.lrs.append(trainer.driver.optimizers[0].param_groups[0]['lr'])


@pytest.mark.parametrize('warmup', [5, 0.1])
@pytest.mark.parametrize('schedule', ['constant', 'linear'])
@pytest.mark.parametrize('accumulation_steps', [1, 3, 4])
def test_torch_warmup_callback(warmup, schedule, accumulation_steps):
    kwargs = get_trainer_args(lr=0.1, bsz=4)
    callback = TorchWarmupCallback(warmup, schedule)
    r_callback = RecordLrCallback()
    kwargs['callbacks'] = [callback, r_callback]
    trainer = Trainer(**kwargs, accumulation_steps=accumulation_steps)
    trainer.run()

    if schedule == 'linear':
        assert kwargs['optimizers'].param_groups[0]['lr'] <= 0.01
    elif schedule == 'constant':
        assert np.allclose(0.1, kwargs['optimizers'].param_groups[0]['lr'])

    assert len(r_callback.lrs)<=trainer.total_batches//accumulation_steps+1
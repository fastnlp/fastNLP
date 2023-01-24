import numpy as np
import pytest

from fastNLP import Trainer
from fastNLP.core.callbacks import Callback, TorchGradClipCallback
from tests.helpers.callbacks.prepare_trainer_args_for_torch_test import \
    get_trainer_args
from tests.helpers.utils import skip_no_cuda


class CheckClipCallback(Callback):

    def __init__(self, parameters, clip_type, clip_value):
        self.parameters = parameters
        self.clip_type = clip_type
        self.clip_value = clip_value

    def on_after_optimizers_step(self, trainer, optimizers):
        for param in self.parameters:
            if self.clip_type == 'value':
                assert param.grad.max().item() <= self.clip_value
            else:
                assert np.linalg.norm(
                    param.grad.cpu().view(-1).numpy()) <= self.clip_value


@pytest.mark.torch
@pytest.mark.parametrize('accumulation_steps', [1, 3, 5])
@pytest.mark.parametrize('fp16', [True, False])
@pytest.mark.parametrize('clip_type', ['norm', 'value'])
@pytest.mark.parametrize('clip_value', [1, 2])
def test_torch_grad_clip_callback(accumulation_steps, fp16, clip_type,
                                  clip_value):
    skip_no_cuda(reason='No cuda, cannot test fp16')
    device = 'cuda' if fp16 else 'cpu'
    kwargs = get_trainer_args(lr=1, device=device)
    callbacks = []
    callbacks.append(
        TorchGradClipCallback(clip_value=clip_value, clip_type=clip_type))
    callbacks.append(
        CheckClipCallback(kwargs['model'].parameters(), clip_type, clip_value))
    trainer = Trainer(**kwargs, callbacks=callbacks, fp16=fp16)
    trainer.run()

import os
from typing import List
import pytest
from dataclasses import dataclass

from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.metrics.accuracy import Accuracy
from fastNLP.core.callbacks.progress_callback import RichCallback
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
from fastNLP.envs.env import USER_CUDA_VISIBLE_DEVICES

if _NEED_IMPORT_PADDLE:
    from paddle.optimizer import Adam
    from paddle.io import DataLoader


from tests.helpers.models.paddle_model import PaddleNormalModel_Classification_1
from tests.helpers.datasets.paddle_data import PaddleRandomMaxDataset
from tests.helpers.utils import magic_argv_env_context

@dataclass
class TrainPaddleConfig:
    num_labels: int = 10
    feature_dimension: int = 10

    batch_size: int = 2
    shuffle: bool = True
    evaluate_every = 2

@pytest.mark.parametrize("device", ["cpu", 1, [0, 1]])
# @pytest.mark.parametrize("driver,device", [("fleet", [0, 1])])
@pytest.mark.parametrize("callbacks", [[RichCallback(5)]])
@pytest.mark.paddledist
@magic_argv_env_context
def test_trainer_paddle(
        device,
        callbacks,
        n_epochs=2,
):
    if isinstance(device, List) and USER_CUDA_VISIBLE_DEVICES not in os.environ:
        pytest.skip("Skip test fleet if FASTNLP_BACKEND is not set to paddle.")
    model = PaddleNormalModel_Classification_1(
        num_labels=TrainPaddleConfig.num_labels,
        feature_dimension=TrainPaddleConfig.feature_dimension
    )
    optimizers = Adam(parameters=model.parameters(), learning_rate=0.0001)
    train_dataloader = DataLoader(
        dataset=PaddleRandomMaxDataset(20, 10),
        batch_size=TrainPaddleConfig.batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=PaddleRandomMaxDataset(20, 10),
        batch_size=TrainPaddleConfig.batch_size,
        shuffle=True
    )
    train_dataloader = train_dataloader
    evaluate_dataloaders = val_dataloader
    evaluate_every = TrainPaddleConfig.evaluate_every
    metrics = {"acc": Accuracy(backend="paddle")}
    trainer = Trainer(
        model=model,
        driver="paddle",
        device=device,
        optimizers=optimizers,
        train_dataloader=train_dataloader,
        evaluate_dataloaders=evaluate_dataloaders,
        evaluate_every=evaluate_every,
        input_mapping=None,
        output_mapping=None,
        metrics=metrics,

        n_epochs=n_epochs,
        callbacks=callbacks,
    )
    trainer.run()

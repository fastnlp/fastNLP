"""
这个文件测试用户以python -m paddle.distributed.launch 启动的情况
看看有没有用pytest执行的机会
python -m paddle.distributed.launch --gpus=0,2,3 test_trainer_fleet.py
"""
import os
os.environ["FASTNLP_BACKEND"] = "paddle"
import sys
sys.path.append("../../../")

from dataclasses import dataclass

from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.metrics.accuracy import Accuracy
from fastNLP.core.callbacks.progress_callback import RichCallback
from fastNLP.core.callbacks import Callback

import paddle
from paddle.optimizer import Adam
from paddle.io import DataLoader

from tests.helpers.models.paddle_model import PaddleNormalModel_Classification_1
from tests.helpers.datasets.paddle_data import PaddleRandomMaxDataset
from tests.helpers.callbacks.helper_callbacks import RecordMetricCallback

@dataclass
class MNISTTrainFleetConfig:
    num_labels: int = 10
    feature_dimension: int = 10

    batch_size: int = 32
    shuffle: bool = True
    validate_every = -1

def test_trainer_fleet(
        driver,
        device,
        callbacks,
        n_epochs,
):
    model = PaddleNormalModel_Classification_1(
        num_labels=MNISTTrainFleetConfig.num_labels,
        feature_dimension=MNISTTrainFleetConfig.feature_dimension
    )
    optimizers = Adam(parameters=model.parameters(), learning_rate=0.0001)

    train_dataloader = DataLoader(
        dataset=PaddleRandomMaxDataset(6400, MNISTTrainFleetConfig.feature_dimension),
        batch_size=MNISTTrainFleetConfig.batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=PaddleRandomMaxDataset(1280, MNISTTrainFleetConfig.feature_dimension),
        batch_size=MNISTTrainFleetConfig.batch_size,
        shuffle=True
    )
    train_dataloader = train_dataloader
    validate_dataloaders = val_dataloader
    validate_every = MNISTTrainFleetConfig.validate_every
    metrics = {"acc": Accuracy()}
    trainer = Trainer(
        model=model,
        driver=driver,
        device=device,
        optimizers=optimizers,
        train_dataloader=train_dataloader,
        evaluate_dataloaders=validate_dataloaders,
        evaluate_every=validate_every,
        input_mapping=None,
        output_mapping=None,
        metrics=metrics,

        n_epochs=n_epochs,
        callbacks=callbacks,
        output_from_new_proc="logs",
    )
    trainer.run()

if __name__ == "__main__":
    driver = "fleet"
    device = [0,2,3]
    # driver = "paddle"
    # device = 2
    callbacks = [
        # RecordMetricCallback(monitor="acc#acc", metric_threshold=0.0, larger_better=True), 
        RichCallback(5),
    ]
    test_trainer_fleet(
        driver=driver,
        device=device,
        callbacks=callbacks,
        n_epochs=5,
    )
"""
这个文件测试用户自己初始化分布式环境后使用 paddle 的情况:

    >>> # 测试用 python -m paddle.distributed.launch 启动
    >>> FASTNLP_BACKEND=paddle python -m paddle.distributed.launch --gpus=0,2,3 _test_trainer_fleet_outside.py
    >>> # 测试在限制 GPU 的情况下用 python -m paddle.distributed.launch 启动
    >>> CUDA_VISIBLE_DEVICES=0,2,3 FASTNLP_BACKEND=paddle python -m paddle.distributed.launch --gpus=0,2,3 _test_trainer_fleet_outside.py

"""
import os
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
import paddle.distributed.fleet as fleet

from tests.helpers.models.paddle_model import PaddleNormalModel_Classification_2
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
    fleet.init(is_collective=True)

    model = PaddleNormalModel_Classification_2(
        num_labels=MNISTTrainFleetConfig.num_labels,
        feature_dimension=MNISTTrainFleetConfig.feature_dimension,
    )
    optimizers = Adam(parameters=model.parameters(), learning_rate=0.0001)

    model = fleet.distributed_model(model)
    optimizers = fleet.distributed_optimizer(optimizers)

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
    data_device = f'gpu:{os.environ["USER_CUDA_VISIBLE_DEVICES"].split(",").index(os.environ["CUDA_VISIBLE_DEVICES"])}'
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
        # output_from_new_proc="logs",
        data_device=data_device
    )
    trainer.run()

if __name__ == "__main__":
    driver = "paddle"
    device = [0,1,3]
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
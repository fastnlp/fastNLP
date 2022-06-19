"""
这个文件测试多卡情况下使用 deepspeed ，且用户自己调用了 deepspeed.initialize 的情况::

    >>> deepspeed _test_trainer_deepspeed_outside.py

"""
import os
import sys
sys.path.append("../../../")
from dataclasses import dataclass

from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.metrics.accuracy import Accuracy
from fastNLP.core.callbacks.progress_callback import RichCallback
from fastNLP.core.drivers.torch_driver.utils import _create_default_config

import deepspeed
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader


from tests.helpers.models.torch_model import TorchNormalModel_Classification_2
from tests.helpers.datasets.torch_data import TorchArgMaxDataset

local_rank = int(os.environ["LOCAL_RANK"])

@dataclass
class TrainDeepSpeedConfig:
    num_labels: int = 3
    feature_dimension: int = 3

    batch_size: int = 2
    shuffle: bool = True
    evaluate_every = 2

def test_trainer_deepspeed(
    device,
    callbacks,
    strategy,
    config,
    n_epochs=2,
):
    model = TorchNormalModel_Classification_2(
        num_labels=TrainDeepSpeedConfig.num_labels,
        feature_dimension=TrainDeepSpeedConfig.feature_dimension
    )
    optimizers = Adam(params=model.parameters(), lr=0.0001)
    train_dataloader = DataLoader(
        dataset=TorchArgMaxDataset(TrainDeepSpeedConfig.feature_dimension, 20),
        batch_size=TrainDeepSpeedConfig.batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=TorchArgMaxDataset(TrainDeepSpeedConfig.feature_dimension, 12),
        batch_size=TrainDeepSpeedConfig.batch_size,
        shuffle=True
    )
    train_dataloader = train_dataloader
    evaluate_dataloaders = val_dataloader
    evaluate_every = TrainDeepSpeedConfig.evaluate_every
    metrics = {"acc": Accuracy()}
    if config is not None:
        config["train_micro_batch_size_per_gpu"] = TrainDeepSpeedConfig.batch_size
    model, optimizers, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizers,
        config=config,
    )
    trainer = Trainer(
        model=model,
        driver="torch",
        device=device,
        data_device=torch.device(f"cuda:{local_rank}"),
        optimizers=optimizers,
        train_dataloader=train_dataloader,
        evaluate_dataloaders=evaluate_dataloaders,
        evaluate_every=evaluate_every,
        metrics=metrics,
        output_mapping={"preds": "pred"},

        n_epochs=n_epochs,
        callbacks=callbacks,
        deepspeed_kwargs={
            "strategy": strategy,
            "config": config
        }
    )
    trainer.run()

if __name__ == "__main__":
    device = [0,1]
    # device = [0,1,3]
    callbacks = [
        # RecordMetricCallback(monitor="acc#acc", metric_threshold=0.0, larger_better=True), 
        RichCallback(5),
    ]
    config = _create_default_config(stage=2)
    test_trainer_deepspeed(
        device=device,
        callbacks=callbacks,
        strategy="deepspeed",
        config=config,
        n_epochs=5,
    )
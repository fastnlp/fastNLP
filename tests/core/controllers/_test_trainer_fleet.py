"""这个文件测试多卡情况下使用 paddle 的情况::

    >>> # 测试用 python -m paddle.distributed.launch 启动
    >>> FASTNLP_BACKEND=paddle python -m paddle.distributed.launch --devices=0,2,3 _test_trainer_fleet.py
    >>> # 测试在限制 GPU 的情况下用 python -m paddle.distributed.launch 启动
    >>> CUDA_VISIBLE_DEVICES=0,2,3 FASTNLP_BACKEND=paddle python -m paddle.distributed.launch --devices=0,2,3 _test_trainer_fleet.py
    >>> # 测试直接使用多卡
    >>> FASTNLP_BACKEND=paddle python _test_trainer_fleet.py
    >>> # 测试在限制 GPU 的情况下直接使用多卡
    >>> CUDA_VISIBLE_DEVICES=3,4,5,6 FASTNLP_BACKEND=paddle python _test_trainer_fleet.py
"""
import os
import sys
import argparse
from dataclasses import dataclass

path = os.path.abspath(__file__)
folders = path.split(os.sep)
for folder in list(folders[::-1]):
    if 'fastnlp' not in folder.lower():
        folders.pop(-1)
    else:
        break
path = os.sep.join(folders)
sys.path.extend([path, os.path.join(path, 'fastNLP')])

from fastNLP.core.callbacks.progress_callback import RichCallback
from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.metrics.accuracy import Accuracy
from tests.helpers.datasets.paddle_data import PaddleArgMaxDataset
from tests.helpers.models.paddle_model import \
    PaddleNormalModel_Classification_1

from paddle.io import DataLoader
from paddle.optimizer import Adam


@dataclass
class MNISTTrainFleetConfig:
    num_labels: int = 5
    feature_dimension: int = 5

    batch_size: int = 4
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
        feature_dimension=MNISTTrainFleetConfig.feature_dimension)
    optimizers = Adam(parameters=model.parameters(), learning_rate=0.0001)

    train_dataloader = DataLoader(
        dataset=PaddleArgMaxDataset(20,
                                    MNISTTrainFleetConfig.feature_dimension),
        batch_size=MNISTTrainFleetConfig.batch_size,
        shuffle=True)
    val_dataloader = DataLoader(
        dataset=PaddleArgMaxDataset(12,
                                    MNISTTrainFleetConfig.feature_dimension),
        batch_size=MNISTTrainFleetConfig.batch_size,
        shuffle=True)
    train_dataloader = train_dataloader
    validate_dataloaders = val_dataloader
    validate_every = MNISTTrainFleetConfig.validate_every
    metrics = {'acc': Accuracy()}
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
        output_from_new_proc='logs',
    )
    trainer.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input trainer parameters.')
    parser.add_argument('-d', '--device', type=int, nargs='+', default=None)

    args = parser.parse_args()
    if args.device is None:
        args.device = [0, 1, 3]

    driver = 'paddle'
    callbacks = [
        # RecordMetricCallback(monitor="acc#acc", metric_threshold=0.0, larger_better=True),
        RichCallback(5),
    ]
    test_trainer_fleet(
        driver=driver,
        device=args.device,
        callbacks=callbacks,
        n_epochs=5,
    )

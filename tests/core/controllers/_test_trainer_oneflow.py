"""
测试 oneflow 动态图的多卡训练::

    >>> # 不使用 DistributedDataParallel 包裹的情况
    >>> python -m oneflow.distributed.launch --nproc_per_node 2 _test_trainer_oneflow.py 
    >>> # 使用 DistributedDataParallel 包裹的情况
    >>> python -m oneflow.distributed.launch --nproc_per_node 2 _test_trainer_oneflow.py -w 
"""
import sys
sys.path.append("../../../")
import os
from dataclasses import dataclass

from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.metrics.accuracy import Accuracy
from fastNLP.envs.imports import _NEED_IMPORT_ONEFLOW

if _NEED_IMPORT_ONEFLOW:
    import oneflow
    from oneflow.nn.parallel import DistributedDataParallel
    from oneflow.optim import Adam
    from oneflow.utils.data import DataLoader

from tests.helpers.models.oneflow_model import OneflowNormalModel_Classification_1
from tests.helpers.datasets.oneflow_data import OneflowArgMaxDataset

@dataclass
class TrainOneflowConfig:
    num_labels: int = 3
    feature_dimension: int = 3

    batch_size: int = 2
    shuffle: bool = True
    evaluate_every = 2

def test_trainer_oneflow(
        callbacks,
        wrapped=False,
        n_epochs=2,
):
    model = OneflowNormalModel_Classification_1(
        num_labels=TrainOneflowConfig.num_labels,
        feature_dimension=TrainOneflowConfig.feature_dimension
    )
    optimizers = Adam(params=model.parameters(), lr=0.0001)
    train_dataloader = DataLoader(
        dataset=OneflowArgMaxDataset(20, TrainOneflowConfig.feature_dimension),
        batch_size=TrainOneflowConfig.batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=OneflowArgMaxDataset(12, TrainOneflowConfig.feature_dimension),
        batch_size=TrainOneflowConfig.batch_size,
        shuffle=True
    )
    train_dataloader = train_dataloader
    evaluate_dataloaders = val_dataloader
    evaluate_every = TrainOneflowConfig.evaluate_every
    metrics = {"acc": Accuracy()}

    if wrapped:
        model.to(int(os.environ["LOCAL_RANK"]))
        model = DistributedDataParallel(model)


    trainer = Trainer(
        model=model,
        driver="oneflow",
        device=0,
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w",
        "--wrapped",
        default=False,
        action="store_true",
        help="Use DistributedDataParallal to wrap model first.",
    )
    args = parser.parse_args()
    
    callbacks = []
    test_trainer_oneflow(callbacks, args.wrapped)

"""

python -m torch.distributed.launch --nproc_per_node 2 tests/core/controllers/_test_distributed_launch_torch_1.py

"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

import sys
path = os.path.abspath(__file__)
folders = path.split(os.sep)
for folder in list(folders[::-1]):
    if 'fastnlp' not in folder.lower():
        folders.pop(-1)
    else:
        break
path = os.sep.join(folders)
sys.path.extend([path, os.path.join(path, 'fastNLP')])

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.optim import SGD
import torch.distributed as dist
from dataclasses import dataclass
from torchmetrics import Accuracy

from fastNLP.core.controllers.trainer import Trainer
from tests.helpers.models.torch_model import TorchNormalModel_Classification_2
from tests.helpers.datasets.torch_data import TorchNormalDataset_Classification


@dataclass
class NormalClassificationTrainTorchConfig:
    num_labels: int = 2
    feature_dimension: int = 3
    each_label_data: int = 100
    seed: int = 0

    n_epochs: int = 10
    batch_size: int = 4
    shuffle: bool = True

    driver: str = "torch"
    device: int = 7


local_rank = int(os.environ["LOCAL_RANK"])
local_device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(local_device)

model = TorchNormalModel_Classification_2(
            num_labels=NormalClassificationTrainTorchConfig.num_labels,
            feature_dimension=NormalClassificationTrainTorchConfig.feature_dimension
        )
model.to(local_device)
dist.init_process_group(backend='nccl', world_size=2, rank=local_rank)
model = DistributedDataParallel(model, device_ids=[local_device.index], output_device=local_device)


dist.barrier()
optimizers = SGD(model.parameters(), lr=0.001)

dataset = TorchNormalDataset_Classification(
    num_labels=NormalClassificationTrainTorchConfig.num_labels,
    feature_dimension=NormalClassificationTrainTorchConfig.feature_dimension,
    each_label_data=NormalClassificationTrainTorchConfig.each_label_data,
    seed=NormalClassificationTrainTorchConfig.seed
)
_dataloader = DataLoader(
    dataset=dataset,
    batch_size=NormalClassificationTrainTorchConfig.batch_size,
    shuffle=True
)
train_dataloader = _dataloader
validate_dataloaders = _dataloader
metrics = {"acc": Accuracy()}


def _test_trainer_torch_with_evaluator_fp16_accumulation_steps(
    accumulation_steps,
    fp16
):
    trainer = Trainer(
        model=model,
        driver="torch_ddp",
        device=None,
        optimizers=optimizers,
        train_dataloader=train_dataloader,
        validate_dataloaders=validate_dataloaders,
        metrics=metrics,

        n_epochs=2,
        data_device=local_device,
        progress_bar='rich',

        accumulation_steps=accumulation_steps,
        fp16=fp16,
    )

    trainer.run()
    # dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Input trainer parameters.')
    parser.add_argument('-v', '--version', type=int, default=0, help="choose one test to run")

    args = parser.parse_args()

    if args.version == 0:
        _test_trainer_torch_with_evaluator_fp16_accumulation_steps(accumulation_steps=1, fp16=False)
    elif args.version == 1:
        _test_trainer_torch_with_evaluator_fp16_accumulation_steps(accumulation_steps=3, fp16=False)
    elif args.version == 2:
        _test_trainer_torch_with_evaluator_fp16_accumulation_steps(accumulation_steps=1, fp16=True)
    elif args.version == 3:
        _test_trainer_torch_with_evaluator_fp16_accumulation_steps(accumulation_steps=3, fp16=True)

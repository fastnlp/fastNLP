import pytest

from fastNLP import Metric, Evaluator

from dataclasses import dataclass
from typing import Any
from itertools import product

from fastNLP.core.controllers.trainer import Trainer
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.datasets.torch_data import TorchNormalDataset_Classification, TorchArgMaxDataset
from tests.helpers.utils import magic_argv_env_context
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from fastNLP import Event

# 检查能否正确 aggregate


class DistMetric(Metric):
    def __init__(self, aggregate_when_get_metric=None):
        super().__init__(aggregate_when_get_metric=aggregate_when_get_metric)
        self.register_element('count', value=0, aggregate_method='sum')
        self.data = 0

    def update(self, y):
        self.count += len(y)
        self.data += len(y)

    def get_metric(self) -> dict:
        count2 = sum(self.all_gather_object(self.data))
        return {'count': self.count.item(), 'count2': count2}

    def reset(self):
        self.data = 0



if _NEED_IMPORT_TORCH:
    from torch.optim import SGD
    from torch.utils.data import DataLoader
    import torch.distributed as dist
    from torch.utils.data import Dataset
    import torch


    class DataSet(Dataset):
        def __init__(self, num_samples=1000, num_features=10):
            g = torch.Generator()
            g.manual_seed(1000)
            self.data = torch.randn(num_samples, num_features, generator=g)
            self.y = self.data.argmax(dim=-1)

        def __getitem__(self, item):
            return {'x': self.data[item], 'y': self.data[item]}

        def __len__(self):
            return len(self.data)


@dataclass
class NormalClassificationTrainTorchConfig:
    num_labels: int = 10
    feature_dimension: int = 10
    seed: int = 0

    batch_size: int = 4
    shuffle: bool = True


@dataclass
class TrainerParameters:
    model: Any = None
    optimizers: Any = None
    train_dataloader: Any = None
    evaluate_dataloaders: Any = None
    input_mapping: Any = None
    output_mapping: Any = None
    metrics: Any = None


@pytest.fixture(scope="module", params=[1], autouse=True)
def trainer_params(request):
    trainer_params = TrainerParameters()

    trainer_params.model = TorchNormalModel_Classification_1(
        num_labels=NormalClassificationTrainTorchConfig.num_labels,
        feature_dimension=NormalClassificationTrainTorchConfig.feature_dimension
    )
    trainer_params.optimizers = SGD(trainer_params.model.parameters(), lr=0.001)

    dataset = DataSet(99, num_features=NormalClassificationTrainTorchConfig.feature_dimension)
    _dataloader = DataLoader(
        dataset=dataset,
        batch_size=NormalClassificationTrainTorchConfig.batch_size,
        shuffle=True
    )
    trainer_params.train_dataloader = _dataloader
    trainer_params.evaluate_dataloaders = _dataloader

    return trainer_params


@pytest.mark.torch
@pytest.mark.parametrize('device', [[0, 1], None])
@magic_argv_env_context
def test_1(trainer_params: TrainerParameters, device):
    # 测试能否自动 aggregate 。
    for aggregate_when_get_metric, use_dist_sampler in product([True, False], [True, False, None]):
        metric = DistMetric(aggregate_when_get_metric=aggregate_when_get_metric)

        evaluator = Evaluator(trainer_params.model, dataloaders=trainer_params.evaluate_dataloaders,
                              metrics={'c': metric},
                              driver='torch', device=device, use_dist_sampler=use_dist_sampler,
                              progress_bar='tqdm')
        if use_dist_sampler is None:
            use_dist_sampler = device is not None
        results = evaluator.run()
        num_samples = len(trainer_params.evaluate_dataloaders.dataset)
        if device is None:
            assert results['count#c'] == num_samples
            assert results['count2#c'] == num_samples
        else:
            if aggregate_when_get_metric is True and use_dist_sampler is True:
                assert results['count#c'] == num_samples
                assert results['count2#c'] == num_samples
            elif aggregate_when_get_metric is True and use_dist_sampler is False:
                assert results['count#c'] == 2*num_samples
                assert results['count2#c'] == 2*num_samples
            elif aggregate_when_get_metric is False and use_dist_sampler is True:
                assert results['count#c'] in (49, 50)  # 不同卡，数量不同
                assert results['count2#c']  in (49, 50)
            else:
                assert results['count#c'] == num_samples
                assert results['count2#c'] == num_samples

    if dist.is_initialized():
        dist.destroy_process_group()



@pytest.mark.torch
@pytest.mark.parametrize('device', [[0, 1], None])
@magic_argv_env_context
def test_2(trainer_params: TrainerParameters, device):
    # 测试能否自动 aggregate 。
    for aggregate_when_get_metric, use_dist_sampler in product([True, False], [True, False, None]):
        metric = DistMetric(aggregate_when_get_metric=aggregate_when_get_metric)

        num_samples = len(trainer_params.evaluate_dataloaders.dataset)

        @Trainer.on(Event.on_sanity_check_end())
        def on_valid_end(trainer, results):
            if device is None:
                assert results['count#c'] == num_samples
                assert results['count2#c'] == num_samples
            else:
                if aggregate_when_get_metric is True and use_dist_sampler is True:
                    assert results['count#c'] == num_samples
                    assert results['count2#c'] == num_samples
                elif aggregate_when_get_metric is True and use_dist_sampler is False:
                    assert results['count#c'] == 2 * num_samples
                    assert results['count2#c'] == 2 * num_samples
                elif aggregate_when_get_metric is False and use_dist_sampler is True:
                    assert results['count#c'] in (49, 50)  # 不同卡，数量不同
                    assert results['count2#c'] in (49, 50)
                else:
                    assert results['count#c'] == num_samples
                    assert results['count2#c'] == num_samples

        trainer = Trainer(
            model=trainer_params.model,
            driver='torch',
            device=device,
            optimizers=trainer_params.optimizers,
            train_dataloader=trainer_params.train_dataloader,
            evaluate_dataloaders=trainer_params.evaluate_dataloaders,
            metrics={'c': metric},
            evaluate_every=-1,
            n_epochs=0,
            output_from_new_proc="all",
            use_dist_sampler=use_dist_sampler,
            progress_bar='tqdm'
        )

        if use_dist_sampler is None:
            use_dist_sampler = device is not None

        trainer.run(num_eval_sanity_batch=-1)

        trainer = Trainer(
            model=trainer_params.model,
            driver='torch',
            device=device,
            optimizers=trainer_params.optimizers,
            train_dataloader=trainer_params.train_dataloader,
            evaluate_dataloaders=trainer_params.evaluate_dataloaders,
            metrics={'c': DistMetric(aggregate_when_get_metric=aggregate_when_get_metric)},
            evaluate_every=-1,
            n_epochs=0,
            output_from_new_proc="all",
            use_dist_sampler=not (use_dist_sampler is True),  #取相反的值
            evaluate_use_dist_sampler=use_dist_sampler,
            progress_bar='rich'  # 刚好测试一下可以替换 progress 么
        )
        trainer.run(num_eval_sanity_batch=-1)

    if dist.is_initialized():
        dist.destroy_process_group()







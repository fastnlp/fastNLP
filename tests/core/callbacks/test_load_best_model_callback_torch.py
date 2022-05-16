import os

from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch
    from torch.utils.data import DataLoader
    from torch import optim
    import torch.distributed as dist

import pytest
from dataclasses import dataclass
from typing import Any
import numpy as np

from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.metrics.accuracy import Accuracy
from fastNLP.core.callbacks.load_best_model_callback import LoadBestModelCallback
from fastNLP.core import Evaluator
from fastNLP.core.drivers.torch_driver import TorchSingleDriver
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.datasets.torch_data import TorchArgMaxDataset
from tests.helpers.utils import magic_argv_env_context
from fastNLP import logger


@dataclass
class ArgMaxDatasetConfig:
    num_labels: int = 10
    feature_dimension: int = 10
    data_num: int = 20
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


@pytest.fixture(scope="module", params=[0], autouse=True)
def model_and_optimizers(request):
    trainer_params = TrainerParameters()

    trainer_params.model = TorchNormalModel_Classification_1(
        num_labels=ArgMaxDatasetConfig.num_labels,
        feature_dimension=ArgMaxDatasetConfig.feature_dimension
    )
    trainer_params.optimizers = optim.SGD(trainer_params.model.parameters(), lr=0.01)
    dataset = TorchArgMaxDataset(
        feature_dimension=ArgMaxDatasetConfig.feature_dimension,
        data_num=ArgMaxDatasetConfig.data_num,
        seed=ArgMaxDatasetConfig.seed
    )
    _dataloader = DataLoader(
        dataset=dataset,
        batch_size=ArgMaxDatasetConfig.batch_size,
        shuffle=True
    )
    trainer_params.train_dataloader = _dataloader
    trainer_params.evaluate_dataloaders = _dataloader
    trainer_params.metrics = {"acc": Accuracy()}

    return trainer_params


@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch", [0, 1]), ("torch", 1), ("torch", "cpu")])  # ("torch", "cpu"), ("torch", [0, 1]), ("torch", 1)
@magic_argv_env_context
def test_load_best_model_callback(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
):
    for save_folder in ['save_models', None]:
        for only_state_dict in [True, False]:
            callbacks = [LoadBestModelCallback(monitor='acc', only_state_dict=only_state_dict,
                                               save_folder=save_folder)]
            trainer = Trainer(
                model=model_and_optimizers.model,
                driver=driver,
                device=device,
                optimizers=model_and_optimizers.optimizers,
                train_dataloader=model_and_optimizers.train_dataloader,
                evaluate_dataloaders=model_and_optimizers.evaluate_dataloaders,
                input_mapping=model_and_optimizers.input_mapping,
                output_mapping=lambda output: output if ('loss' in output) else {'pred':output['preds'], 'target': output['target']},
                metrics={'acc': Accuracy()},
                n_epochs=2,
                callbacks=callbacks,
                output_from_new_proc="all"
            )

            trainer.run(num_eval_sanity_batch=0)

            _driver = TorchSingleDriver(model_and_optimizers.model, device=torch.device('cuda'))
            evaluator = Evaluator(model_and_optimizers.model, driver=_driver, device=device,
                                  dataloaders={'dl1': model_and_optimizers.evaluate_dataloaders},
                                  metrics={'acc': Accuracy(aggregate_when_get_metric=False)},
                                  output_mapping=lambda output: output if ('loss' in output) else {'pred':output['preds'], 'target': output['target']},
                                  progress_bar='rich', use_dist_sampler=False)
            results = evaluator.run()
            assert np.allclose(callbacks[0].monitor_value, results['acc#acc#dl1'])
            trainer.driver.barrier()
            if save_folder:
                import shutil
                shutil.rmtree(save_folder, ignore_errors=True)
    if dist.is_initialized():
        dist.destroy_process_group()

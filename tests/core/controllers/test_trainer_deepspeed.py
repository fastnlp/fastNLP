import pytest
from dataclasses import dataclass

from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.metrics.accuracy import Accuracy
from fastNLP.core.callbacks.progress_callback import RichCallback
from fastNLP.core.drivers.torch_driver import DeepSpeedDriver
from fastNLP.core.drivers.torch_driver.utils import _create_default_config
from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    import torch
    from torch.optim import Adam
    from torch.utils.data import DataLoader


from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.datasets.torch_data import TorchArgMaxDataset
from tests.helpers.utils import magic_argv_env_context

@dataclass
class TrainDeepSpeedConfig:
    num_labels: int = 3
    feature_dimension: int = 3

    batch_size: int = 2
    shuffle: bool = True
    evaluate_every = 2

@pytest.mark.deepspeed
class TestTrainer:
    @classmethod
    def setup_class(cls):
        # 不初始化的话从第二个测试例开始会因为环境变量报错。
        torch_model = TorchNormalModel_Classification_1(1, 1)
        torch_opt = torch.optim.Adam(params=torch_model.parameters(), lr=0.01)
        device = [torch.device(i) for i in [0,1]]
        driver = DeepSpeedDriver(
            model=torch_model,
            parallel_device=device,
        )
        driver.set_optimizers(torch_opt)
        driver.setup()

        return driver

    @pytest.mark.parametrize("device", [[0, 1]])
    @pytest.mark.parametrize("callbacks", [[RichCallback(5)]])
    @pytest.mark.parametrize("strategy", ["deepspeed", "deepspeed_stage_1"])
    @pytest.mark.parametrize("config", [None, _create_default_config(stage=1)])
    @magic_argv_env_context
    def test_trainer_deepspeed(
        self,
        device,
        callbacks,
        strategy,
        config,
        n_epochs=2,
    ):
        model = TorchNormalModel_Classification_1(
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
        trainer = Trainer(
            model=model,
            driver="torch",
            device=device,
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

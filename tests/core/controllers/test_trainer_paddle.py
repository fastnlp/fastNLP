import pytest
import os
from typing import Any
from dataclasses import dataclass

from paddle.optimizer import Adam
from paddle.io import DataLoader

from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.metrics.accuracy import Accuracy
from fastNLP.core.callbacks.progress_callback import RichCallback
from fastNLP.envs import FASTNLP_DISTRIBUTED_CHECK


from tests.helpers.models.paddle_model import PaddleNormalModel_Classification
from tests.helpers.datasets.paddle_data import PaddleDataset_MNIST
from tests.helpers.callbacks.helper_callbacks import RecordLossCallback, RecordMetricCallback
from tests.helpers.utils import magic_argv_env_context

@dataclass
class MNISTTrainPaddleConfig:
    num_labels: int = 10
    feature_dimension: int = 784

    batch_size: int = 32
    shuffle: bool = True
    validate_every = -5

    driver: str = "paddle"
    device = "gpu"

@dataclass
class MNISTTrainFleetConfig:
    num_labels: int = 10
    feature_dimension: int = 784

    batch_size: int = 32
    shuffle: bool = True
    validate_every = -5

@dataclass
class TrainerParameters:
    model: Any = None
    optimizers: Any = None
    train_dataloader: Any = None
    validate_dataloaders: Any = None
    input_mapping: Any = None
    output_mapping: Any = None
    metrics: Any = None

# @pytest.fixture(params=[0], autouse=True)
# def model_and_optimizers(request):
#     """
#     初始化单卡模式的模型和优化器
#     """
#     trainer_params = TrainerParameters()
#     print(paddle.device.get_device())

#     if request.param == 0:
#         trainer_params.model = PaddleNormalModel_Classification(
#             num_labels=MNISTTrainPaddleConfig.num_labels,
#             feature_dimension=MNISTTrainPaddleConfig.feature_dimension
#         )
#         trainer_params.optimizers = Adam(parameters=trainer_params.model.parameters(), learning_rate=0.0001)
#         train_dataloader = DataLoader(
#             dataset=PaddleDataset_MNIST("train"),
#             batch_size=MNISTTrainPaddleConfig.batch_size,
#             shuffle=True
#         )
#         val_dataloader = DataLoader(
#             dataset=PaddleDataset_MNIST(mode="test"),
#             batch_size=MNISTTrainPaddleConfig.batch_size,
#             shuffle=True
#         )
#         trainer_params.train_dataloader = train_dataloader
#         trainer_params.validate_dataloaders = val_dataloader
#         trainer_params.validate_every = MNISTTrainPaddleConfig.validate_every
#         trainer_params.metrics = {"acc": Accuracy()}

#     return trainer_params


@pytest.mark.parametrize("driver,device", [("paddle", "cpu"), ("paddle", 1)])
# @pytest.mark.parametrize("driver,device", [("fleet", [0, 1])])
@pytest.mark.parametrize("callbacks", [[RecordMetricCallback(monitor="acc#acc", metric_threshold=0.7, larger_better=True), 
                                        RichCallback(5), RecordLossCallback(loss_threshold=0.3)]])
@magic_argv_env_context
def test_trainer_paddle(
        # model_and_optimizers: TrainerParameters,
        driver,
        device,
        callbacks,
        n_epochs=15,
):
    trainer_params = TrainerParameters()

    trainer_params.model = PaddleNormalModel_Classification(
        num_labels=MNISTTrainPaddleConfig.num_labels,
        feature_dimension=MNISTTrainPaddleConfig.feature_dimension
    )
    trainer_params.optimizers = Adam(parameters=trainer_params.model.parameters(), learning_rate=0.0001)
    train_dataloader = DataLoader(
        dataset=PaddleDataset_MNIST("train"),
        batch_size=MNISTTrainPaddleConfig.batch_size,
        shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=PaddleDataset_MNIST(mode="test"),
        batch_size=MNISTTrainPaddleConfig.batch_size,
        shuffle=True
    )
    trainer_params.train_dataloader = train_dataloader
    trainer_params.validate_dataloaders = val_dataloader
    trainer_params.validate_every = MNISTTrainPaddleConfig.validate_every
    trainer_params.metrics = {"acc": Accuracy(backend="paddle")}
    if not isinstance(device, (int, str)) and len(device) > 1 and FASTNLP_DISTRIBUTED_CHECK not in os.environ:
        with pytest.raises(SystemExit) as exc:
            trainer = Trainer(
                model=trainer_params.model,
                driver=driver,
                device=device,
                optimizers=trainer_params.optimizers,
                train_dataloader=trainer_params.train_dataloader,
                validate_dataloaders=trainer_params.validate_dataloaders,
                validate_every=trainer_params.validate_every,
                input_mapping=trainer_params.input_mapping,
                output_mapping=trainer_params.output_mapping,
                metrics=trainer_params.metrics,

                n_epochs=n_epochs,
                callbacks=callbacks,
            )
        assert exc.value.code == 0
        return
    else:
        trainer = Trainer(
            model=trainer_params.model,
            driver=driver,
            device=device,
            optimizers=trainer_params.optimizers,
            train_dataloader=trainer_params.train_dataloader,
            validate_dataloaders=trainer_params.validate_dataloaders,
            validate_every=trainer_params.validate_every,
            input_mapping=trainer_params.input_mapping,
            output_mapping=trainer_params.output_mapping,
            metrics=trainer_params.metrics,

            n_epochs=n_epochs,
            callbacks=callbacks,
        )
        trainer.run()
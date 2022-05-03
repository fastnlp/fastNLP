import pytest
from typing import Any
from dataclasses import dataclass
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
import torch.distributed as dist

from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.callbacks.callback_event import Event
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.datasets.torch_data import TorchNormalDataset_Classification
from tests.helpers.callbacks.helper_callbacks import RecordTrainerEventTriggerCallback
from tests.helpers.utils import magic_argv_env_context, Capturing


@dataclass
class NormalClassificationTrainTorchConfig:
    num_labels: int = 2
    feature_dimension: int = 3
    each_label_data: int = 100
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


@pytest.fixture(scope="module", autouse=True)
def model_and_optimizers():
    trainer_params = TrainerParameters()

    trainer_params.model = TorchNormalModel_Classification_1(
        num_labels=NormalClassificationTrainTorchConfig.num_labels,
        feature_dimension=NormalClassificationTrainTorchConfig.feature_dimension
    )
    trainer_params.optimizers = SGD(trainer_params.model.parameters(), lr=0.001)
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
    trainer_params.train_dataloader = _dataloader
    trainer_params.evaluate_dataloaders = _dataloader
    trainer_params.metrics = {"acc": Accuracy()}

    return trainer_params


@pytest.mark.parametrize("driver,device", [("torch", "cpu")])  # , ("torch", 6), ("torch", [6, 7])
@pytest.mark.parametrize("callbacks", [[RecordTrainerEventTriggerCallback()]])
@pytest.mark.torch
@magic_argv_env_context
def test_trainer_event_trigger_1(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        callbacks,
        n_epochs=2,
):

    with pytest.raises(Exception):
        with Capturing() as output:
            trainer = Trainer(
                model=model_and_optimizers.model,
                driver=driver,
                device=device,
                optimizers=model_and_optimizers.optimizers,
                train_dataloader=model_and_optimizers.train_dataloader,
                evaluate_dataloaders=model_and_optimizers.evaluate_dataloaders,
                input_mapping=model_and_optimizers.input_mapping,
                output_mapping=model_and_optimizers.output_mapping,
                metrics=model_and_optimizers.metrics,

                n_epochs=n_epochs,
                callbacks=callbacks
            )

            trainer.run()

            if dist.is_initialized():
                dist.destroy_process_group()

            Event_attrs = Event.__dict__
            for k, v in Event_attrs.items():
                if isinstance(v, staticmethod):
                    assert k in output[0]

@pytest.mark.parametrize("driver,device", [("torch", "cpu")])  # , ("torch", 6), ("torch", [6, 7])
@pytest.mark.torch
@magic_argv_env_context
def test_trainer_event_trigger_2(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        n_epochs=2,
):

    @Trainer.on(Event.on_after_trainer_initialized())
    def on_after_trainer_initialized(trainer, driver):
        print("on_after_trainer_initialized")

    @Trainer.on(Event.on_sanity_check_begin())
    def on_sanity_check_begin(trainer):
        print("on_sanity_check_begin")

    @Trainer.on(Event.on_sanity_check_end())
    def on_sanity_check_end(trainer, sanity_check_res):
        print("on_sanity_check_end")

    @Trainer.on(Event.on_train_begin())
    def on_train_begin(trainer):
        print("on_train_begin")

    @Trainer.on(Event.on_train_end())
    def on_train_end(trainer):
        print("on_train_end")

    @Trainer.on(Event.on_train_epoch_begin())
    def on_train_epoch_begin(trainer):
        if trainer.cur_epoch_idx >= 1:
            # 触发 on_exception；
            raise Exception
        print("on_train_epoch_begin")

    @Trainer.on(Event.on_train_epoch_end())
    def on_train_epoch_end(trainer):
        print("on_train_epoch_end")

    @Trainer.on(Event.on_fetch_data_begin())
    def on_fetch_data_begin(trainer):
        print("on_fetch_data_begin")

    @Trainer.on(Event.on_fetch_data_end())
    def on_fetch_data_end(trainer):
        print("on_fetch_data_end")

    @Trainer.on(Event.on_train_batch_begin())
    def on_train_batch_begin(trainer, batch, indices=None):
        print("on_train_batch_begin")

    @Trainer.on(Event.on_train_batch_end())
    def on_train_batch_end(trainer):
        print("on_train_batch_end")

    @Trainer.on(Event.on_exception())
    def on_exception(trainer, exception):
        print("on_exception")

    @Trainer.on(Event.on_before_backward())
    def on_before_backward(trainer, outputs):
        print("on_before_backward")

    @Trainer.on(Event.on_after_backward())
    def on_after_backward(trainer):
        print("on_after_backward")

    @Trainer.on(Event.on_before_optimizers_step())
    def on_before_optimizers_step(trainer, optimizers):
        print("on_before_optimizers_step")

    @Trainer.on(Event.on_after_optimizers_step())
    def on_after_optimizers_step(trainer, optimizers):
        print("on_after_optimizers_step")

    @Trainer.on(Event.on_before_zero_grad())
    def on_before_zero_grad(trainer, optimizers):
        print("on_before_zero_grad")

    @Trainer.on(Event.on_after_zero_grad())
    def on_after_zero_grad(trainer, optimizers):
        print("on_after_zero_grad")

    @Trainer.on(Event.on_evaluate_begin())
    def on_evaluate_begin(trainer):
        print("on_evaluate_begin")

    @Trainer.on(Event.on_evaluate_end())
    def on_evaluate_end(trainer, results):
        print("on_evaluate_end")

    with pytest.raises(Exception):
        with Capturing() as output:
            trainer = Trainer(
                model=model_and_optimizers.model,
                driver=driver,
                device=device,
                optimizers=model_and_optimizers.optimizers,
                train_dataloader=model_and_optimizers.train_dataloader,
                evaluate_dataloaders=model_and_optimizers.evaluate_dataloaders,
                input_mapping=model_and_optimizers.input_mapping,
                output_mapping=model_and_optimizers.output_mapping,
                metrics=model_and_optimizers.metrics,

                n_epochs=n_epochs,
            )

            trainer.run()
        Event_attrs = Event.__dict__
        for k, v in Event_attrs.items():
            if isinstance(v, staticmethod):
                assert k in output[0]

@pytest.mark.parametrize("driver,device", [("torch", "cpu")])  # , ("torch", 6), ("torch", [6, 7])
@pytest.mark.torch
@magic_argv_env_context
def test_trainer_event_trigger_3(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        n_epochs=2,
):

    @Trainer.on(Event.on_after_trainer_initialized())
    def on_after_trainer_initialized(trainer, driver):
        print("on_after_trainer_initialized")

    @Trainer.on(Event.on_sanity_check_begin())
    def on_sanity_check_begin(trainer):
        print("on_sanity_check_begin")

    @Trainer.on(Event.on_sanity_check_end())
    def on_sanity_check_end(trainer, sanity_check_res):
        print("on_sanity_check_end")

    @Trainer.on(Event.on_train_begin())
    def on_train_begin(trainer):
        print("on_train_begin")

    @Trainer.on(Event.on_train_end())
    def on_train_end(trainer):
        print("on_train_end")

    @Trainer.on(Event.on_train_epoch_begin())
    def on_train_epoch_begin(trainer):
        if trainer.cur_epoch_idx >= 1:
            # 触发 on_exception；
            raise Exception
        print("on_train_epoch_begin")

    @Trainer.on(Event.on_train_epoch_end())
    def on_train_epoch_end(trainer):
        print("on_train_epoch_end")

    @Trainer.on(Event.on_fetch_data_begin())
    def on_fetch_data_begin(trainer):
        print("on_fetch_data_begin")

    @Trainer.on(Event.on_fetch_data_end())
    def on_fetch_data_end(trainer):
        print("on_fetch_data_end")

    @Trainer.on(Event.on_train_batch_begin())
    def on_train_batch_begin(trainer, batch, indices=None):
        print("on_train_batch_begin")

    @Trainer.on(Event.on_train_batch_end())
    def on_train_batch_end(trainer):
        print("on_train_batch_end")

    @Trainer.on(Event.on_exception())
    def on_exception(trainer, exception):
        print("on_exception")

    @Trainer.on(Event.on_before_backward())
    def on_before_backward(trainer, outputs):
        print("on_before_backward")

    @Trainer.on(Event.on_after_backward())
    def on_after_backward(trainer):
        print("on_after_backward")

    @Trainer.on(Event.on_before_optimizers_step())
    def on_before_optimizers_step(trainer, optimizers):
        print("on_before_optimizers_step")

    @Trainer.on(Event.on_after_optimizers_step())
    def on_after_optimizers_step(trainer, optimizers):
        print("on_after_optimizers_step")

    @Trainer.on(Event.on_before_zero_grad())
    def on_before_zero_grad(trainer, optimizers):
        print("on_before_zero_grad")

    @Trainer.on(Event.on_after_zero_grad())
    def on_after_zero_grad(trainer, optimizers):
        print("on_after_zero_grad")

    @Trainer.on(Event.on_evaluate_begin())
    def on_evaluate_begin(trainer):
        print("on_evaluate_begin")

    @Trainer.on(Event.on_evaluate_end())
    def on_evaluate_end(trainer, results):
        print("on_evaluate_end")

    with pytest.raises(Exception):
        with Capturing() as output:
            trainer = Trainer(
                model=model_and_optimizers.model,
                driver=driver,
                device=device,
                optimizers=model_and_optimizers.optimizers,
                train_dataloader=model_and_optimizers.train_dataloader,
                evaluate_dataloaders=model_and_optimizers.evaluate_dataloaders,
                input_mapping=model_and_optimizers.input_mapping,
                output_mapping=model_and_optimizers.output_mapping,
                metrics=model_and_optimizers.metrics,

                n_epochs=n_epochs,
            )

            trainer.run()

        Event_attrs = Event.__dict__
        for k, v in Event_attrs.items():
            if isinstance(v, staticmethod):
                assert k in output[0]

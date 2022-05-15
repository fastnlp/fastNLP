import os.path
import subprocess
import sys
import pytest

from dataclasses import dataclass
from typing import Any
from pathlib import Path

from fastNLP.core.controllers.trainer import Trainer
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1, TorchNormalModel_Classification_3
from tests.helpers.datasets.torch_data import TorchNormalDataset_Classification
from tests.helpers.callbacks.helper_callbacks import RecordLossCallback
from tests.helpers.callbacks.helper_callbacks_torch import RecordAccumulationStepsCallback_Torch
from tests.helpers.utils import magic_argv_env_context, Capturing
from fastNLP.envs.distributed import rank_zero_rm
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch.distributed as dist
    from torch.optim import SGD
    from torch.utils.data import DataLoader


@dataclass
class NormalClassificationTrainTorchConfig:
    num_labels: int = 2
    feature_dimension: int = 3
    each_label_data: int = 10
    seed: int = 0

    n_epochs: int = 3
    batch_size: int = 4
    shuffle: bool = True

    driver: str = "torch"
    device: int = 1


@dataclass
class TrainerParameters:
    model: Any = None
    optimizers: Any = None
    train_dataloader: Any = None
    evaluate_dataloaders: Any = None
    input_mapping: Any = None
    output_mapping: Any = None
    metrics: Any = None


@pytest.fixture(scope="function", params=[0], autouse=True)
def model_and_optimizers(request):
    trainer_params = TrainerParameters()

    if request.param == 0:
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
        trainer_params.train_dataloader = DataLoader(
            dataset=dataset,
            batch_size=NormalClassificationTrainTorchConfig.batch_size,
            shuffle=True
        )
        trainer_params.evaluate_dataloaders = None
        trainer_params.input_mapping = None
        trainer_params.output_mapping = None

    # elif request.param == 1:


    return trainer_params


# 测试一下 cpu；
@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch", "cpu")])
@magic_argv_env_context
def test_trainer_torch_without_evaluator(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        n_epochs=3,
):
    callbacks = [RecordLossCallback(loss_threshold=0.5)]
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
        callbacks=callbacks,

    )

    trainer.run()

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch", 1), ("torch", [1, 2])])  # ("torch", 4),
@pytest.mark.parametrize("fp16", [False, True])
@pytest.mark.parametrize("accumulation_steps", [1, 3])
@magic_argv_env_context
def test_trainer_torch_without_evaluator_fp16_accumulation_steps(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        fp16,
        accumulation_steps,
        n_epochs=3,
):
    callbacks = [RecordLossCallback(loss_threshold=0.5)]
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
        callbacks=callbacks,
        fp16=fp16,
        accumulation_steps=accumulation_steps,

        output_from_new_proc="all"
    )

    trainer.run()

    if dist.is_initialized():
        dist.destroy_process_group()


# 测试 accumulation_steps；
@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch", "cpu"), ("torch", 1), ("torch", [1, 2])])
@pytest.mark.parametrize("accumulation_steps", [1, 3])
@magic_argv_env_context
def test_trainer_torch_without_evaluator_accumulation_steps(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        accumulation_steps,
        n_epochs=2,
):
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
        callbacks=[RecordAccumulationStepsCallback_Torch(accumulation_steps)],
        accumulation_steps=accumulation_steps
    )

    trainer.run()

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch", [1, 2])])
@pytest.mark.parametrize("output_from_new_proc", ["all", "ignore", "only_error", "test_log"])
@magic_argv_env_context
def test_trainer_output_from_new_proc(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        output_from_new_proc,
        n_epochs=2,
):
    std_msg = "test std msg trainer, std std std"
    err_msg = "test err msg trainer, err err, err"

    from fastNLP.core.log.logger import logger

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

            output_from_new_proc=output_from_new_proc
        )

        if trainer.driver.get_local_rank() != 0:
            logger.warning(std_msg)
            sys.stderr.write(err_msg)

        trainer.run()

        if dist.is_initialized():
            dist.destroy_process_group()

    if output_from_new_proc == "all":
        if trainer.driver.get_local_rank() != 0:
            assert std_msg in output[0]
            assert err_msg in output[0]
    elif output_from_new_proc == "ignore":
        if trainer.driver.get_local_rank() != 0:
            assert std_msg not in output[0]
            assert err_msg not in output[0]
    elif output_from_new_proc == "only_error":
        if trainer.driver.get_local_rank() != 0:
            assert std_msg not in output[0]
            assert err_msg in output[0]
    else:
        std_path = Path(os.path.abspath(output_from_new_proc)).joinpath(f"{trainer.driver.get_local_rank()}_std.log")
        assert std_path.exists()
        err_path = Path(os.path.abspath(output_from_new_proc)).joinpath(f"{trainer.driver.get_local_rank()}_err.log")
        assert err_path.exists()

        path = Path(os.path.abspath(output_from_new_proc))
        rank_zero_rm(path)


@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch", [0, 1])])
@pytest.mark.parametrize("cur_rank", [0])  # 依次测试如果是当前进程出现错误，是否能够正确地 kill 掉其他进程；  , 1, 2, 3
@magic_argv_env_context
def test_trainer_on_exception(
        model_and_optimizers: TrainerParameters,
        driver,
        device,
        cur_rank,
        n_epochs=2,
):
    from fastNLP.core.callbacks.callback_event import Event

    @Trainer.on(Event.on_train_epoch_end())
    def raise_exception(trainer):
        if trainer.driver.get_local_rank() == cur_rank:
            raise NotImplementedError

    with pytest.raises(NotImplementedError):
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
            output_from_new_proc="all"
        )
        trainer.run()

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.torch
@pytest.mark.parametrize("version", [0, 1, 2, 3])
@magic_argv_env_context
def test_torch_distributed_launch_1(version):
    """
    测试用户自己在外面初始化 ddp；
    """
    from fastNLP.core.drivers.torch_driver.ddp import find_free_network_port
    path = Path(os.path.abspath(__file__)).parent
    command = ["python", "-m", "torch.distributed.launch", "--nproc_per_node", "2", "--master_port", find_free_network_port(),
               f"{path.joinpath('_test_distributed_launch_torch_1.py')}", "-v", f"{version}"]
    subprocess.check_call(command, env=os.environ)


@pytest.mark.torch
@pytest.mark.parametrize("version", [0, 1, 2, 3])
@magic_argv_env_context
def test_torch_distributed_launch_2(version):
    """
    测试用户自己不初始化 ddp，但是使用 torch.distributed.launch 启动；
    """
    from fastNLP.core.drivers.torch_driver.ddp import find_free_network_port
    path = Path(os.path.abspath(__file__)).parent
    command = ["python", "-m", "torch.distributed.launch", "--nproc_per_node", "2", "--master_port", find_free_network_port(),
               f"{path.joinpath('_test_distributed_launch_torch_2.py')}", "-v", f"{version}"]
    subprocess.check_call(command, env=os.environ)


@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch", 0), ("torch", [0, 1])])
@magic_argv_env_context
def test_torch_wo_auto_param_call(
    driver,
    device,
    n_epochs=2,
):

    model = TorchNormalModel_Classification_3(
        num_labels=NormalClassificationTrainTorchConfig.num_labels,
        feature_dimension=NormalClassificationTrainTorchConfig.feature_dimension
    )
    optimizers = SGD(model.parameters(), lr=0.001)
    dataset = TorchNormalDataset_Classification(
        num_labels=NormalClassificationTrainTorchConfig.num_labels,
        feature_dimension=NormalClassificationTrainTorchConfig.feature_dimension,
        each_label_data=NormalClassificationTrainTorchConfig.each_label_data,
        seed=NormalClassificationTrainTorchConfig.seed
    )
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=NormalClassificationTrainTorchConfig.batch_size,
        shuffle=True
    )

    trainer = Trainer(
        model=model,
        driver=driver,
        device=device,
        optimizers=optimizers,
        train_dataloader=train_dataloader,
        n_epochs=n_epochs,

        model_wo_auto_param_call=True,
        output_from_new_proc="all"
    )

    trainer.run()

    if dist.is_initialized():
        dist.destroy_process_group()





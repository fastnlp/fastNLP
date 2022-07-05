import os
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import re

import pytest
from fastNLP.core.controllers.trainer import Trainer
from torchmetrics import Accuracy
from fastNLP.core.callbacks import CheckpointCallback
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.datasets.torch_data import TorchNormalDataset_Classification, TorchArgMaxDataset
from tests.helpers.callbacks.helper_callbacks import RecordLossCallback
from tests.helpers.utils import magic_argv_env_context
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from fastNLP.envs import FASTNLP_LAUNCH_TIME, rank_zero_rm
if _NEED_IMPORT_TORCH:
    import torch.distributed as dist
    from torch.optim import SGD
    from torch.utils.data import DataLoader


@dataclass
class ArgMaxDatasetConfig:
    num_labels: int = 10
    feature_dimension: int = 10
    data_num: int = 50
    seed: int = 0

    batch_size: int = 2
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
    trainer_params.optimizers = SGD(trainer_params.model.parameters(), lr=0.001)
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
@magic_argv_env_context
def test_trainer_torch_without_evaluator(
        model_and_optimizers: TrainerParameters,
        n_epochs=3,
):
    callbacks = [RecordLossCallback(loss_threshold=0.5)]
    trainer = Trainer(
        model=model_and_optimizers.model,
        driver="torch_fsdp",
        device=[4, 5],
        optimizers=model_and_optimizers.optimizers,
        train_dataloader=model_and_optimizers.train_dataloader,
        evaluate_dataloaders=model_and_optimizers.evaluate_dataloaders,
        input_mapping=model_and_optimizers.input_mapping,
        output_mapping=model_and_optimizers.output_mapping,
        metrics=model_and_optimizers.metrics,

        n_epochs=3,
        callbacks=callbacks,
        output_from_new_proc="all"

    )

    trainer.run()

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch_fsdp", [4, 5])])
@magic_argv_env_context(timeout=100)
def test_model_checkpoint_callback_1(
    model_and_optimizers: TrainerParameters,
    driver,
    device
):
    for version in [0]:
        # 需要在每一个循环开始重新初始化 model，是因为 fsdp 会将当前卡上的 model 删除，从而导致这个引用实际上引用到的是一个空模型；
        model_and_optimizers.model = TorchNormalModel_Classification_1(
            num_labels=ArgMaxDatasetConfig.num_labels,
            feature_dimension=ArgMaxDatasetConfig.feature_dimension
        )
        try:
            path = Path.cwd().joinpath(f"test_model_checkpoint")
            path.mkdir(exist_ok=True, parents=True)

            if version == 0:
                callbacks = [
                    CheckpointCallback(folder=path, every_n_epochs=1, every_n_batches=123, last=False, on_exceptions=None, topk=0,
                                       monitor=None, only_state_dict=True, save_object='model')
                ]
            elif version == 1:
                callbacks = [
                    CheckpointCallback(folder=path, every_n_epochs=3, every_n_batches=None, last=True, on_exceptions=None, topk=2,
                                       monitor="acc", only_state_dict=True, save_object='model')
                ]

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
                n_epochs=10,
                callbacks=callbacks,
                output_from_new_proc="all",
                # torch_kwargs={"fsdp_kwargs": {'save_on_rank0': True}}
            )

            trainer.run()
            print("Finish train")
            all_saved_model_paths = {w.name: w for w in path.joinpath(os.environ[FASTNLP_LAUNCH_TIME]).iterdir()}
            # 检查生成保存模型文件的数量是不是正确的；
            if version == 0:

                if not isinstance(device, list):
                    assert "model-epoch_10" in all_saved_model_paths
                    assert "model-epoch_4-batch_123" in all_saved_model_paths

                    epoch_save_path = all_saved_model_paths["model-epoch_10"]
                    step_save_path = all_saved_model_paths["model-epoch_4-batch_123"]

                    assert len(all_saved_model_paths) == 12
                # ddp 下的文件名不同，因为同样的数据，ddp 用了更少的步数跑完；
                else:
                    assert "model-epoch_6" in all_saved_model_paths
                    assert "model-epoch_9-batch_123" in all_saved_model_paths

                    epoch_save_path = all_saved_model_paths["model-epoch_6"]
                    step_save_path = all_saved_model_paths["model-epoch_9-batch_123"]

                    assert len(all_saved_model_paths) == 11
                all_state_dicts = [epoch_save_path]#, step_save_path]

            elif version == 1:

                pattern = re.compile("model-epoch_[0-9]+-batch_[0-9]+-[a-zA-Z#]+_[0-9]*.?[0-9]*")

                if not isinstance(device, list):
                    assert "model-epoch_9" in all_saved_model_paths
                    assert "model-last" in all_saved_model_paths
                    aLL_topk_folders = []
                    for each_folder_name in all_saved_model_paths:
                        each_folder_name = pattern.findall(each_folder_name)
                        if len(each_folder_name) != 0:
                            aLL_topk_folders.append(each_folder_name[0])
                    assert len(aLL_topk_folders) == 2

                    epoch_save_path = all_saved_model_paths["model-epoch_9"]
                    last_save_path = all_saved_model_paths["model-last"]
                    topk_save_path = all_saved_model_paths[aLL_topk_folders[0]]

                    assert len(all_saved_model_paths) == 6
                # ddp 下的文件名不同，因为同样的数据，ddp 用了更少的步数跑完；
                else:
                    assert "model-epoch_9" in all_saved_model_paths
                    assert "model-last" in all_saved_model_paths

                    aLL_topk_folders = []
                    for each_folder_name in all_saved_model_paths:
                        each_folder_name = pattern.findall(each_folder_name)
                        if len(each_folder_name) != 0:
                            aLL_topk_folders.append(each_folder_name[0])
                    assert len(aLL_topk_folders) == 2

                    epoch_save_path = all_saved_model_paths["model-epoch_9"]
                    last_save_path = all_saved_model_paths["model-last"]
                    topk_save_path = all_saved_model_paths[aLL_topk_folders[0]]

                    assert len(all_saved_model_paths) == 6

                all_state_dicts = [epoch_save_path, last_save_path, topk_save_path]

            for folder in all_state_dicts:
                model_and_optimizers.model = TorchNormalModel_Classification_1(
                    num_labels=ArgMaxDatasetConfig.num_labels,
                    feature_dimension=ArgMaxDatasetConfig.feature_dimension
                )

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

                    n_epochs=20,
                    output_from_new_proc="all",

                )
                trainer.load_model(folder, only_state_dict=True)

                trainer.run()
                trainer.driver.barrier()
        finally:
            rank_zero_rm(path)

    if dist.is_initialized():
        dist.destroy_process_group()





@pytest.mark.skip("现在 fsdp 还不支持断点重训；")
@pytest.mark.torch
@pytest.mark.parametrize("driver,device", [("torch_fsdp", [6, 7])])  # ("torch", "cpu"), ("torch", [0, 1]), ("torch", 1)
@magic_argv_env_context(timeout=100)
def test_trainer_checkpoint_callback_1(
    model_and_optimizers: TrainerParameters,
    driver,
    device
):
    for version in [0, 1]:
        model_and_optimizers.model = TorchNormalModel_Classification_1(
            num_labels=ArgMaxDatasetConfig.num_labels,
            feature_dimension=ArgMaxDatasetConfig.feature_dimension
        )
        try:
            path = Path.cwd().joinpath(f"test_model_checkpoint")
            path.mkdir(exist_ok=True, parents=True)

            if version == 0:
                callbacks = [
                    CheckpointCallback(folder=path, every_n_epochs=7, every_n_batches=123, last=False, on_exceptions=None, topk=0,
                                       monitor=None, only_state_dict=True, save_object='trainer')
                ]
            elif version == 1:
                callbacks = [
                    CheckpointCallback(folder=path, every_n_epochs=None, every_n_batches=None, last=True, on_exceptions=None,
                                       topk=2, monitor="acc", only_state_dict=True, save_object='trainer')
                ]

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

                n_epochs=10,
                callbacks=callbacks,
                output_from_new_proc="all"
            )

            trainer.run()

            all_saved_model_paths = {w.name: w for w in path.joinpath(os.environ[FASTNLP_LAUNCH_TIME]).iterdir()}
            # 检查生成保存模型文件的数量是不是正确的；
            if version == 0:

                if not isinstance(device, list):
                    assert "trainer-epoch_7" in all_saved_model_paths
                    assert "trainer-epoch_4-batch_123" in all_saved_model_paths

                    epoch_save_path = all_saved_model_paths["trainer-epoch_7"]
                    step_save_path = all_saved_model_paths["trainer-epoch_4-batch_123"]

                    assert len(all_saved_model_paths) == 3
                # ddp 下的文件名不同，因为同样的数据，ddp 用了更少的步数跑完；
                else:
                    assert "trainer-epoch_7" in all_saved_model_paths
                    assert "trainer-epoch_9-batch_123" in all_saved_model_paths

                    epoch_save_path = all_saved_model_paths["trainer-epoch_7"]
                    step_save_path = all_saved_model_paths["trainer-epoch_9-batch_123"]

                    assert len(all_saved_model_paths) == 2
                all_state_dicts = [epoch_save_path, step_save_path]

            elif version == 1:

                pattern = re.compile("trainer-epoch_[0-9]+-batch_[0-9]+-[a-zA-Z#]+_[0-9]*.?[0-9]*")

                # all_saved_model_paths = {w.name: w for w in path.joinpath(os.environ[FASTNLP_LAUNCH_TIME]).iterdir()}
                if not isinstance(device, list):
                    assert "trainer-last" in all_saved_model_paths
                    aLL_topk_folders = []
                    for each_folder_name in all_saved_model_paths:
                        each_folder_name = pattern.findall(each_folder_name)
                        if len(each_folder_name) != 0:
                            aLL_topk_folders.append(each_folder_name[0])
                    assert len(aLL_topk_folders) == 2

                    last_save_path = all_saved_model_paths["trainer-last"]
                    topk_save_path = all_saved_model_paths[aLL_topk_folders[0]]

                    assert len(all_saved_model_paths) == 3
                # ddp 下的文件名不同，因为同样的数据，ddp 用了更少的步数跑完；
                else:
                    assert "trainer-last" in all_saved_model_paths

                    aLL_topk_folders = []
                    for each_folder_name in all_saved_model_paths:
                        each_folder_name = pattern.findall(each_folder_name)
                        if len(each_folder_name) != 0:
                            aLL_topk_folders.append(each_folder_name[0])
                    assert len(aLL_topk_folders) == 2

                    last_save_path = all_saved_model_paths["trainer-last"]
                    topk_save_path = all_saved_model_paths[aLL_topk_folders[0]]

                    assert len(all_saved_model_paths) == 3

                all_state_dicts = [last_save_path, topk_save_path]

            for folder in all_state_dicts:
                model_and_optimizers.model = TorchNormalModel_Classification_1(
                    num_labels=ArgMaxDatasetConfig.num_labels,
                    feature_dimension=ArgMaxDatasetConfig.feature_dimension
                )

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

                    n_epochs=13,
                    output_from_new_proc="all"
                )
                trainer.load_checkpoint(folder, only_state_dict=True)

                trainer.run()
                trainer.driver.barrier()

        finally:
            rank_zero_rm(path)

    if dist.is_initialized():
        dist.destroy_process_group()
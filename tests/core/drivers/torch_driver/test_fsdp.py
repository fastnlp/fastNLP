import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from fastNLP.core.callbacks import CheckpointCallback
from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.drivers.torch_driver import TorchFSDPDriver
from fastNLP.core.samplers import RandomSampler
from fastNLP.envs import FASTNLP_LAUNCH_TIME, rank_zero_rm
from fastNLP.envs.imports import _NEED_IMPORT_TORCH, _TORCH_GREATER_EQUAL_1_12
from tests.helpers.callbacks.helper_callbacks import RecordLossCallback
from tests.helpers.datasets.torch_data import (TorchArgMaxDataset,
                                               TorchNormalXYDataset)
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.utils import magic_argv_env_context, skip_no_cuda

if _NEED_IMPORT_TORCH:
    import torch
    import torch.distributed as dist
    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from torchmetrics import Accuracy


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


@pytest.fixture(scope='module', params=[0], autouse=True)
def model_and_optimizers(request):
    trainer_params = TrainerParameters()

    trainer_params.model = TorchNormalModel_Classification_1(
        num_labels=ArgMaxDatasetConfig.num_labels,
        feature_dimension=ArgMaxDatasetConfig.feature_dimension)
    trainer_params.optimizers = Adam(
        trainer_params.model.parameters(), lr=0.001)
    dataset = TorchArgMaxDataset(
        feature_dimension=ArgMaxDatasetConfig.feature_dimension,
        data_num=ArgMaxDatasetConfig.data_num,
        seed=ArgMaxDatasetConfig.seed)
    _dataloader = DataLoader(
        dataset=dataset,
        batch_size=ArgMaxDatasetConfig.batch_size,
        shuffle=True)
    trainer_params.train_dataloader = _dataloader
    trainer_params.evaluate_dataloaders = _dataloader
    trainer_params.metrics = {
        'acc':
        Accuracy(
            task='multiclass',
            num_classes=ArgMaxDatasetConfig.feature_dimension)
    }

    return trainer_params


def generate_driver(labels,
                    features,
                    device=[0, 1],
                    fp16=False,
                    output_from_new_proc='all'):
    torch_model = TorchNormalModel_Classification_1(labels, features)
    torch_opt = Adam(params=torch_model.parameters(), lr=0.001)
    device = [torch.device(i) for i in device]
    driver = TorchFSDPDriver(
        model=torch_model,
        parallel_device=device,
        fp16=fp16,
        output_from_new_proc=output_from_new_proc)
    driver.set_optimizers(torch_opt)
    driver.setup()

    return driver


def dataloader_with_randomsampler(dataset,
                                  batch_size,
                                  shuffle,
                                  drop_last,
                                  seed=0):
    """建立一个 sampler 为 RandomSampler 的 dataloader."""
    sampler = RandomSampler(dataset, shuffle, seed=seed)
    dataloader = DataLoader(
        dataset, sampler=sampler, drop_last=drop_last, batch_size=batch_size)
    return dataloader


def compare_dict(d1, d2):
    assert len(d1) == len(d2)
    for k in d1.keys():
        assert k in d2
        v1 = d1[k]
        v2 = d2[k]
        assert type(v1) is type(v2)
        if isinstance(v1, dict):
            compare_dict(v1, v2)
        elif isinstance(v1, torch.Tensor):
            assert torch.equal(v1, v2)
        else:
            assert v1 == v2


@pytest.mark.skipif(
    not _TORCH_GREATER_EQUAL_1_12, reason='fsdp 需要 torch 版本在 1.12 及以上')
@pytest.mark.torch
@pytest.mark.parametrize('device', [[0, 1]])
@pytest.mark.parametrize('on_rank0', [True, False])
@pytest.mark.parametrize('optim_shard_strategy', ['shard', 'scatter'])
@magic_argv_env_context
def test_driver_checkpoint(device, on_rank0, optim_shard_strategy):
    skip_no_cuda()
    path = 'fsdp_ckpt'
    try:
        num_replicas = len(device)
        driver1 = generate_driver(100, 1, device)
        driver2 = generate_driver(100, 1, device)
        dataset = TorchNormalXYDataset(100)

        dataloader = dataloader_with_randomsampler(dataset, 4, True, False)
        dataloader.batch_sampler.sampler.set_distributed(
            num_replicas=driver1.world_size,
            rank=driver1.global_rank,
            pad=True)
        num_consumed_batches = 4

        already_seen_x_set = set()
        already_seen_y_set = set()
        driver1.set_sampler_epoch(dataloader, 4)
        for idx, batch in enumerate(dataloader):
            if idx >= num_consumed_batches:
                break
            already_seen_x_set.update(batch['x'].reshape(-1, ).tolist())
            already_seen_y_set.update(batch['y'].reshape(-1, ).tolist())
            res1 = driver1.model(
                batch,
                fastnlp_fn=driver1.model.module.model.train_step,
                fastnlp_signature_fn=None,
                wo_auto_param_call=False,
            )
            driver1.backward(res1['loss'])
            driver1.zero_grad()
            driver1.step()

        # 同步
        dist.barrier()

        # 保存状态
        sampler_states = dataloader.batch_sampler.sampler.state_dict()
        save_states = {'num_consumed_batches': num_consumed_batches}
        driver1.save_checkpoint(
            Path(path),
            save_states,
            dataloader,
            only_state_dict=True,
            should_save_model=True,
            on_rank0=on_rank0)
        dist.barrier()  # 等待save成功

        # 加载
        # 更改 batch_size
        dataloader = dataloader_with_randomsampler(dataset, 2, True, False)
        dataloader.batch_sampler.sampler.set_distributed(
            num_replicas=driver2.world_size,
            rank=driver2.global_rank,
            pad=True)
        load_states = driver2.load_checkpoint(
            Path(path),
            dataloader,
            only_state_dict=True,
            should_load_model=True,
            on_rank0=on_rank0,
            optim_shard_strategy=optim_shard_strategy)
        replaced_loader = load_states.pop('dataloader')

        # 1. 检查 optimizer 的状态
        optim_state_1 = driver1.get_optimizer_state(True)
        optim_state_2 = driver2.get_optimizer_state(True)
        compare_dict(optim_state_1, optim_state_2)

        # 2. 检查 sampler 是否被正确地加载和替换
        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
        if os.environ['FASTNLP_GLOBAL_RANK'] == '0':
            assert replaced_loader.batch_sampler.sampler.seed == sampler_states[
                'seed']
            assert replaced_loader.batch_sampler.sampler.epoch == sampler_states[
                'epoch']
            assert len(replaced_loader.batch_sampler.sampler.dataset
                       ) == sampler_states['length']
            assert replaced_loader.batch_sampler.sampler.shuffle == sampler_states[
                'shuffle']
        assert replaced_loader.batch_sampler.sampler.num_consumed_samples == 4 * num_consumed_batches * num_replicas

        # 4. 检查 model 的参数是否正确
        # 5. 检查 batch_idx
        start_batch = load_states.pop('batch_idx_in_epoch')
        assert start_batch == 2 * num_consumed_batches
        left_x_batches = set()
        left_y_batches = set()
        driver2.set_sampler_epoch(replaced_loader, 4)
        for idx, batch in enumerate(replaced_loader):

            left_x_batches.update(batch['x'].reshape(-1, ).tolist())
            left_y_batches.update(batch['y'].reshape(-1, ).tolist())
            res1 = driver1.model(
                batch,
                fastnlp_fn=driver1.model.module.model.evaluate_step,
                fastnlp_signature_fn=None,
                wo_auto_param_call=False,
            )
            res2 = driver2.model(
                batch,
                fastnlp_fn=driver2.model.module.model.evaluate_step,
                fastnlp_signature_fn=None,
                wo_auto_param_call=False,
            )
            assert torch.equal(res1['preds'], res2['preds'])

        assert len(left_x_batches) + len(
            already_seen_x_set) == len(dataset) / num_replicas
        assert len(left_x_batches
                   | already_seen_x_set) == len(dataset) / num_replicas
        assert len(left_y_batches) + len(
            already_seen_y_set) == len(dataset) / num_replicas
        assert len(left_y_batches
                   | already_seen_y_set) == len(dataset) / num_replicas

    finally:
        rank_zero_rm(path)

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.skipif(
    not _TORCH_GREATER_EQUAL_1_12, reason='fsdp 需要 torch 版本在 1.12 及以上')
@pytest.mark.torch
@magic_argv_env_context
def test_trainer_torch_without_evaluator(
        model_and_optimizers: TrainerParameters):
    skip_no_cuda()
    callbacks = [RecordLossCallback(loss_threshold=0.5)]
    trainer = Trainer(
        model=model_and_optimizers.model,
        driver='torch_fsdp',
        device=[0, 1],
        optimizers=model_and_optimizers.optimizers,
        train_dataloader=model_and_optimizers.train_dataloader,
        evaluate_dataloaders=model_and_optimizers.evaluate_dataloaders,
        input_mapping=model_and_optimizers.input_mapping,
        output_mapping=model_and_optimizers.output_mapping,
        metrics=model_and_optimizers.metrics,
        n_epochs=3,
        callbacks=callbacks,
        output_from_new_proc='all')

    trainer.run()

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.skipif(
    not _TORCH_GREATER_EQUAL_1_12, reason='fsdp 需要 torch 版本在 1.12 及以上')
@pytest.mark.torch
@pytest.mark.parametrize('on_rank0', [True, False])
@magic_argv_env_context(timeout=100)
def test_model_checkpoint_callback_1(model_and_optimizers: TrainerParameters,
                                     on_rank0):
    skip_no_cuda()
    device = [0, 1]
    for version in [0, 1]:
        # 需要在每一个循环开始重新初始化 model，是因为 fsdp 会将当前卡上的 model 删除，从而导致这个引用实际上引用到的是一个空模型
        model_and_optimizers.model = TorchNormalModel_Classification_1(
            num_labels=ArgMaxDatasetConfig.num_labels,
            feature_dimension=ArgMaxDatasetConfig.feature_dimension)
        try:
            path = Path.cwd().joinpath('test_model_checkpoint')
            path.mkdir(exist_ok=True, parents=True)

            if version == 0:
                callbacks = [
                    CheckpointCallback(
                        folder=path,
                        every_n_epochs=1,
                        every_n_batches=123,
                        last=False,
                        on_exceptions=None,
                        topk=0,
                        monitor=None,
                        only_state_dict=True,
                        save_object='model',
                        on_rank0=on_rank0,
                    )
                ]
            elif version == 1:
                callbacks = [
                    CheckpointCallback(
                        folder=path,
                        every_n_epochs=3,
                        every_n_batches=None,
                        last=True,
                        on_exceptions=None,
                        topk=2,
                        monitor='acc',
                        only_state_dict=True,
                        save_object='model',
                        on_rank0=on_rank0)
                ]

            trainer = Trainer(
                model=model_and_optimizers.model,
                driver='torch_fsdp',
                device=device,
                optimizers=model_and_optimizers.optimizers,
                train_dataloader=model_and_optimizers.train_dataloader,
                evaluate_dataloaders=model_and_optimizers.evaluate_dataloaders,
                input_mapping=model_and_optimizers.input_mapping,
                output_mapping=model_and_optimizers.output_mapping,
                metrics=model_and_optimizers.metrics,
                n_epochs=10,
                callbacks=callbacks,
                output_from_new_proc='all',
            )

            trainer.run()
            print('Finish train')
            all_saved_model_paths = {
                w.name: w
                for w in path.joinpath(
                    os.environ[FASTNLP_LAUNCH_TIME]).iterdir()
            }
            # 检查生成保存模型文件的数量是不是正确的；
            if version == 0:

                if not isinstance(device, list):
                    assert 'model-epoch_10' in all_saved_model_paths
                    assert 'model-epoch_4-batch_123' in all_saved_model_paths

                    epoch_save_path = all_saved_model_paths['model-epoch_10']
                    step_save_path = all_saved_model_paths[
                        'model-epoch_4-batch_123']

                    assert len(all_saved_model_paths) == 12
                # ddp 下的文件名不同，因为同样的数据，ddp 用了更少的步数跑完；
                else:
                    assert 'model-epoch_6' in all_saved_model_paths
                    assert 'model-epoch_9-batch_123' in all_saved_model_paths

                    epoch_save_path = all_saved_model_paths['model-epoch_6']
                    step_save_path = all_saved_model_paths[
                        'model-epoch_9-batch_123']

                    assert len(all_saved_model_paths) == 11
                all_state_dicts = [epoch_save_path, step_save_path]

            elif version == 1:

                pattern = re.compile(
                    'model-epoch_[0-9]+-batch_[0-9]+-[a-zA-Z#]+_[0-9]*.?[0-9]*'
                )

                if not isinstance(device, list):
                    assert 'model-epoch_9' in all_saved_model_paths
                    assert 'model-last' in all_saved_model_paths
                    aLL_topk_folders = []
                    for each_folder_name in all_saved_model_paths:
                        each_folder_name = pattern.findall(each_folder_name)
                        if len(each_folder_name) != 0:
                            aLL_topk_folders.append(each_folder_name[0])
                    assert len(aLL_topk_folders) == 2

                    epoch_save_path = all_saved_model_paths['model-epoch_9']
                    last_save_path = all_saved_model_paths['model-last']
                    topk_save_path = all_saved_model_paths[aLL_topk_folders[0]]

                    assert len(all_saved_model_paths) == 6
                # ddp 下的文件名不同，因为同样的数据，ddp 用了更少的步数跑完；
                else:
                    assert 'model-epoch_9' in all_saved_model_paths
                    assert 'model-last' in all_saved_model_paths

                    aLL_topk_folders = []
                    for each_folder_name in all_saved_model_paths:
                        each_folder_name = pattern.findall(each_folder_name)
                        if len(each_folder_name) != 0:
                            aLL_topk_folders.append(each_folder_name[0])
                    assert len(aLL_topk_folders) == 2

                    epoch_save_path = all_saved_model_paths['model-epoch_9']
                    last_save_path = all_saved_model_paths['model-last']
                    topk_save_path = all_saved_model_paths[aLL_topk_folders[0]]

                    assert len(all_saved_model_paths) == 6

                all_state_dicts = [
                    epoch_save_path, last_save_path, topk_save_path
                ]

            for folder in all_state_dicts:
                model_and_optimizers.model = TorchNormalModel_Classification_1(
                    num_labels=ArgMaxDatasetConfig.num_labels,
                    feature_dimension=ArgMaxDatasetConfig.feature_dimension)

                trainer = Trainer(
                    model=model_and_optimizers.model,
                    driver='torch_fsdp',
                    device=device,
                    optimizers=model_and_optimizers.optimizers,
                    train_dataloader=model_and_optimizers.train_dataloader,
                    evaluate_dataloaders=model_and_optimizers.
                    evaluate_dataloaders,
                    input_mapping=model_and_optimizers.input_mapping,
                    output_mapping=model_and_optimizers.output_mapping,
                    metrics=model_and_optimizers.metrics,
                    n_epochs=2,
                    output_from_new_proc='all',
                )
                trainer.load_model(
                    folder, only_state_dict=True, on_rank0=on_rank0)

                trainer.run()
                trainer.driver.barrier()
        finally:
            rank_zero_rm(path)
            pass

    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.mark.torch
@pytest.mark.skipif(
    not _TORCH_GREATER_EQUAL_1_12, reason='fsdp 需要 torch 版本在 1.12 及以上')
@pytest.mark.parametrize('driver,device', [('torch_fsdp', [0, 1])])
@pytest.mark.parametrize('on_rank0', [True, False])
@pytest.mark.parametrize('optim_shard_strategy', ['shard'])
@magic_argv_env_context(timeout=100)
def test_trainer_checkpoint_callback_1(model_and_optimizers: TrainerParameters,
                                       driver, device, on_rank0,
                                       optim_shard_strategy):
    skip_no_cuda(device)
    for version in [0, 1]:
        model_and_optimizers.model = TorchNormalModel_Classification_1(
            num_labels=ArgMaxDatasetConfig.num_labels,
            feature_dimension=ArgMaxDatasetConfig.feature_dimension)
        try:
            path = Path.cwd().joinpath('test_model_checkpoint')
            path.mkdir(exist_ok=True, parents=True)

            if version == 0:
                callbacks = [
                    CheckpointCallback(
                        folder=path,
                        every_n_epochs=7,
                        every_n_batches=123,
                        last=False,
                        on_exceptions=None,
                        topk=0,
                        monitor=None,
                        only_state_dict=True,
                        save_object='trainer',
                        on_rank0=on_rank0,
                        optim_shard_strategy=optim_shard_strategy,
                    )
                ]
            elif version == 1:
                callbacks = [
                    CheckpointCallback(
                        folder=path,
                        every_n_epochs=None,
                        every_n_batches=None,
                        last=True,
                        on_exceptions=None,
                        topk=2,
                        monitor='acc',
                        only_state_dict=True,
                        save_object='trainer',
                        on_rank0=on_rank0,
                        optim_shard_strategy=optim_shard_strategy)
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
                output_from_new_proc='all')

            trainer.run()

            all_saved_model_paths = {
                w.name: w
                for w in path.joinpath(
                    os.environ[FASTNLP_LAUNCH_TIME]).iterdir()
            }
            # 检查生成保存模型文件的数量是不是正确的；
            if version == 0:

                if not isinstance(device, list):
                    assert 'trainer-epoch_7' in all_saved_model_paths
                    assert 'trainer-epoch_4-batch_123' in all_saved_model_paths

                    epoch_save_path = all_saved_model_paths['trainer-epoch_7']
                    step_save_path = all_saved_model_paths[
                        'trainer-epoch_4-batch_123']

                    assert len(all_saved_model_paths) == 3
                # ddp 下的文件名不同，因为同样的数据，ddp 用了更少的步数跑完；
                else:
                    assert 'trainer-epoch_7' in all_saved_model_paths
                    assert 'trainer-epoch_9-batch_123' in all_saved_model_paths

                    epoch_save_path = all_saved_model_paths['trainer-epoch_7']
                    step_save_path = all_saved_model_paths[
                        'trainer-epoch_9-batch_123']

                    assert len(all_saved_model_paths) == 2
                all_state_dicts = [epoch_save_path, step_save_path]

            elif version == 1:

                pattern = re.compile('trainer-epoch_[0-9]+-batch_[0-9]+' +
                                     '-[a-zA-Z#]+_[0-9]*.?[0-9]*')

                all_saved_model_paths = {
                    w.name: w
                    for w in path.joinpath(
                        os.environ[FASTNLP_LAUNCH_TIME]).iterdir()
                }
                if not isinstance(device, list):
                    assert 'trainer-last' in all_saved_model_paths
                    aLL_topk_folders = []
                    for each_folder_name in all_saved_model_paths:
                        each_folder_name = pattern.findall(each_folder_name)
                        if len(each_folder_name) != 0:
                            aLL_topk_folders.append(each_folder_name[0])
                    assert len(aLL_topk_folders) == 2

                    last_save_path = all_saved_model_paths['trainer-last']
                    topk_save_path = all_saved_model_paths[aLL_topk_folders[0]]

                    assert len(all_saved_model_paths) == 3
                # ddp 下的文件名不同，因为同样的数据，ddp 用了更少的步数跑完；
                else:
                    assert 'trainer-last' in all_saved_model_paths

                    aLL_topk_folders = []
                    for each_folder_name in all_saved_model_paths:
                        each_folder_name = pattern.findall(each_folder_name)
                        if len(each_folder_name) != 0:
                            aLL_topk_folders.append(each_folder_name[0])
                    assert len(aLL_topk_folders) == 2

                    last_save_path = all_saved_model_paths['trainer-last']
                    topk_save_path = all_saved_model_paths[aLL_topk_folders[0]]

                    assert len(all_saved_model_paths) == 3

                all_state_dicts = [last_save_path, topk_save_path]

            for folder in all_state_dicts:
                model_and_optimizers.model = TorchNormalModel_Classification_1(
                    num_labels=ArgMaxDatasetConfig.num_labels,
                    feature_dimension=ArgMaxDatasetConfig.feature_dimension)

                trainer = Trainer(
                    model=model_and_optimizers.model,
                    driver=driver,
                    device=device,
                    optimizers=model_and_optimizers.optimizers,
                    train_dataloader=model_and_optimizers.train_dataloader,
                    evaluate_dataloaders=model_and_optimizers.
                    evaluate_dataloaders,
                    input_mapping=model_and_optimizers.input_mapping,
                    output_mapping=model_and_optimizers.output_mapping,
                    metrics=model_and_optimizers.metrics,
                    n_epochs=13,
                    output_from_new_proc='all')
                trainer.load_checkpoint(
                    folder,
                    only_state_dict=True,
                    on_rank0=on_rank0,
                    optim_shard_strategy=optim_shard_strategy)

                trainer.run()
                trainer.driver.barrier()

        finally:
            rank_zero_rm(path)

    if dist.is_initialized():
        dist.destroy_process_group()

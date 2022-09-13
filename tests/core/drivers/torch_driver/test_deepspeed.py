import os
from pathlib import Path

import pytest

from fastNLP.core.drivers.torch_driver.deepspeed import DeepSpeedDriver
from fastNLP.core.samplers import (
    RandomSampler,
    BucketedBatchSampler,
    UnrepeatedRandomSampler,
)
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.datasets.torch_data import TorchNormalXYDataset
from tests.helpers.utils import magic_argv_env_context
from fastNLP.envs.distributed import rank_zero_rm
from fastNLP import logger
from fastNLP.envs.imports import _NEED_IMPORT_TORCH, _NEED_IMPORT_DEEPSPEED

if _NEED_IMPORT_TORCH:
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    
if _NEED_IMPORT_DEEPSPEED:
    import deepspeed

def generate_driver(labels, features, device=[0,1], fp16=False, output_from_new_proc="all", train_dataloader=None):
    torch_model = TorchNormalModel_Classification_1(labels, features)
    torch_opt = torch.optim.Adam(params=torch_model.parameters(), lr=0.01)
    device = [torch.device(i) for i in device]
    driver = DeepSpeedDriver(
        model=torch_model,
        parallel_device=device,
        fp16=fp16,
        output_from_new_proc=output_from_new_proc,
        train_dataloader=train_dataloader
    )
    driver.set_optimizers(torch_opt)
    driver.setup()

    return driver

def dataloader_with_bucketedbatchsampler(dataset, length, batch_size, shuffle, drop_last):
    """
    建立一个 batch_sampler 为 BucketedBatchSampler 的 dataloader
    """
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=BucketedBatchSampler(
            dataset,
            length,
            batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        ),
    )

    return dataloader

def dataloader_with_randomsampler(dataset, batch_size, shuffle, drop_last, seed=0, unrepeated=False):
    """
    建立一个 sampler 为 RandomSampler 的 dataloader
    """
    if unrepeated:
        sampler = UnrepeatedRandomSampler(dataset, shuffle, seed)
    else:
        sampler = RandomSampler(dataset, shuffle, seed=seed)
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        drop_last=drop_last,
        batch_size=batch_size
    )
    return dataloader

############################################################################
#
# 测试 TorchDeepSpeedDriver 的一些函数
#
############################################################################

# @pytest.mark.deepspeed
# @magic_argv_env_context
# def test_multi_drivers():
#     """
#     测试使用了多个 TorchDeepSpeedDriver 的情况。
#     """
#     generate_driver(10, 10)
#     generate_driver(20, 10)
    
#     with pytest.raises(RuntimeError):
#         # 设备设置不同，应该报错
#         generate_driver(20, 3, device=[0,1,2])
#         assert False
#     dist.barrier()

#     if dist.is_initialized():
#         dist.destroy_process_group()

@pytest.mark.deepspeed
@magic_argv_env_context
def test_multi_optimizers():
    torch_model = TorchNormalModel_Classification_1(10, 10)
    torch_opt = torch.optim.Adam(params=torch_model.parameters(), lr=0.01)
    device = [torch.device(i) for i in [0, 1]]
    driver = DeepSpeedDriver(
        model=torch_model,
        parallel_device=device,
    )
    driver.set_optimizers([torch_opt, torch_opt])
    with pytest.raises(ValueError):
        driver.setup()

    # if dist.is_initialized():
    #     dist.destroy_process_group()

@pytest.mark.deepspeed
class TestDeepSpeedDriverFunction:
    """
    测试 TorchDeepSpeedDriver 一些简单函数的测试类，基本都是测试能否运行、是否存在 import 错误等问题
    """
    @classmethod
    def setup_class(cls):
        cls.driver = generate_driver(10, 10)

    @magic_argv_env_context
    def test_simple_functions(self):
        """
        简单测试多个函数
        """

        """
        测试 move_data_to_device 函数。这个函数仅调用了 torch_move_data_to_device ，测试例在
        tests/core/utils/test_torch_utils.py中，就不重复测试了
        """
        self.driver.move_data_to_device(torch.rand((32, 64)))
        dist.barrier()

        """
        测试 is_distributed 函数
        """
        assert self.driver.is_distributed() == True
        dist.barrier()

        """
        测试 get_no_sync_context 函数
        """
        res = self.driver.get_model_no_sync_context()
        dist.barrier()

        """
        测试 is_global_zero 函数
        """
        self.driver.is_global_zero()
        dist.barrier()

        """
        测试 unwrap_model 函数
        """
        self.driver.unwrap_model()
        dist.barrier()

        """
        测试 get_local_rank 函数
        """
        self.driver.get_local_rank()
        dist.barrier()

        """
        测试 all_gather 函数
        详细的测试在 test_dist_utils.py 中完成
        """
        obj = {
            "rank": self.driver.global_rank
        }
        obj_list = self.driver.all_gather(obj, group=None)
        for i, res in enumerate(obj_list):
            assert res["rank"] == i

        """
        测试 broadcast_object 函数
        详细的函数在 test_dist_utils.py 中完成
        """
        if self.driver.global_rank == 0:
            obj = {
                "rank": self.driver.global_rank
            }
        else:
            obj = None
        res = self.driver.broadcast_object(obj, src=0)
        assert res["rank"] == 0

        # if dist.is_initialized():
        #     dist.destroy_process_group()

############################################################################
#
# 测试 save 和 load 相关的功能
#
############################################################################
@pytest.mark.deepspeed
class TestSaveLoad:
    """
    测试多卡情况下 save 和 load 相关函数的表现
    """
    @classmethod
    def setup_class(cls):
        # 不在这里 setup 的话会报错
        cls.driver = generate_driver(10, 10, device=[0,1])

    def setup_method(self):
        self.dataset = TorchNormalXYDataset(100)

    @magic_argv_env_context
    @pytest.mark.parametrize("only_state_dict", ([True, False]))
    def test_save_and_load_model(self, only_state_dict):
        """
        测试 save_model 和 load_model 函数
        """
        try:
            path = "model"

            dataloader = DataLoader(self.dataset, batch_size=2)
            driver1, driver2 = generate_driver(20, 1, train_dataloader=dataloader), \
                                generate_driver(20, 1, train_dataloader=dataloader)

            driver1.save_model(path, only_state_dict)

            # 同步
            dist.barrier()
            driver2.load_model(path, only_state_dict)

            for idx, batch in enumerate(dataloader):
                batch = driver1.move_data_to_device(batch)
                res1 = driver1.model(
                    batch,
                    fastnlp_fn=driver1.model.module.model.evaluate_step,
                    # Driver.model -> DataParallel.module -> _FleetWrappingModel.model
                    fastnlp_signature_fn=None,
                    wo_auto_param_call=False,
                )
                res2 = driver2.model(
                    batch,
                    fastnlp_fn=driver2.model.module.model.evaluate_step,
                    fastnlp_signature_fn=None,
                    wo_auto_param_call=False,
                )

                assert torch.equal(res1["preds"], res2["preds"])
        finally:
            rank_zero_rm(path)

        # if dist.is_initialized():
        #     dist.destroy_process_group()

    @magic_argv_env_context
    @pytest.mark.parametrize("only_state_dict", ([True, False]))
    @pytest.mark.parametrize("fp16", ([True, False]))
    @pytest.mark.parametrize("device", ([[0,1]]))
    def test_save_and_load_with_bucketedbatchsampler(self, device, only_state_dict, fp16):
        """
        测试save和load函数，主要测试 dataloader 被替换了 sampler 之后的情况
        """

        try:
            path = "model.ckp"
            num_replicas = len(device)

            dataloader = dataloader_with_bucketedbatchsampler(
                self.dataset,
                length=[10 for i in range(len(self.dataset))],
                batch_size=4,
                shuffle=True,
                drop_last=False
            )
            dataloader.batch_sampler.set_distributed(
                num_replicas=int(os.getenv("WORLD_SIZE", "1")),
                rank=int(os.getenv("RANK", "0")),
                pad=True,
            )
            num_consumed_batches = 4
            driver1, driver2 = generate_driver(20, 1, device=device, fp16=fp16, train_dataloader=dataloader), \
                                            generate_driver(20, 1, device=device, fp16=False, train_dataloader=dataloader)

            already_seen_x_set = set()
            already_seen_y_set = set()
            driver1.set_sampler_epoch(dataloader, 4)
            for idx, batch in enumerate(dataloader):
                if idx >= num_consumed_batches:
                    break
                already_seen_x_set.update(batch["x"].reshape(-1, ).tolist())
                already_seen_y_set.update(batch["y"].reshape(-1, ).tolist())

            # 同步
            dist.barrier()

            # 保存状态
            sampler_states = dataloader.batch_sampler.state_dict()
            save_states = {"num_consumed_batches": num_consumed_batches}
            driver1.save_checkpoint(Path(path), save_states, dataloader, only_state_dict, should_save_model=True)
            dist.barrier()
            # 加载
            # 更改 batch_size
            dataloader = dataloader_with_bucketedbatchsampler(
                self.dataset,
                length=[10 for i in range(len(self.dataset))],
                batch_size=2,
                shuffle=True,
                drop_last=False
            )
            dataloader.batch_sampler.set_distributed(
                num_replicas=driver2.world_size,
                rank=driver2.global_rank,
                pad=True
            )
            dist.barrier()
            load_states = driver2.load_checkpoint(Path(path), dataloader, only_state_dict, should_load_model=True)
            dist.barrier()
            replaced_loader = load_states.pop("dataloader")
            
            # 1. 检查 optimizer 的状态
            # TODO optimizer 的 state_dict 总是为空

            # 2. 检查 batch_sampler 是否被正确地加载和替换
            assert not (replaced_loader is dataloader)
            assert replaced_loader.batch_sampler is dataloader.batch_sampler
            assert isinstance(replaced_loader.batch_sampler, BucketedBatchSampler)
            if os.environ['FASTNLP_GLOBAL_RANK'] == '0':
                assert replaced_loader.batch_sampler.seed == sampler_states["seed"]
            assert replaced_loader.batch_sampler.num_consumed_samples == num_consumed_batches * 4 * num_replicas

            # 4. 检查 model 的参数是否正确
            # 5. 检查 batch_idx
            start_batch = load_states.pop('batch_idx_in_epoch')
            assert start_batch == 2 * num_consumed_batches
            left_x_batches = set()
            left_y_batches = set()
            driver2.set_sampler_epoch(replaced_loader, 4)
            for idx, batch in enumerate(replaced_loader):

                left_x_batches.update(batch["x"].reshape(-1, ).tolist())
                left_y_batches.update(batch["y"].reshape(-1, ).tolist())
                batch = driver1.move_data_to_device(batch)
                res1 = driver1.model(
                    batch,
                    fastnlp_fn=driver1.model.module.model.evaluate_step,
                    # Driver.model -> DataParallel.module -> _FleetWrappingModel.model
                    fastnlp_signature_fn=None,
                    wo_auto_param_call=False,
                )
                res2 = driver2.model(
                    batch,
                    fastnlp_fn=driver2.model.module.model.evaluate_step,
                    fastnlp_signature_fn=None,
                    wo_auto_param_call=False,
                )
                assert torch.equal(res1["preds"], res2["preds"])

            assert len(left_x_batches) + len(already_seen_x_set) == len(self.dataset) / num_replicas
            assert len(left_x_batches | already_seen_x_set) == len(self.dataset) / num_replicas
            assert len(left_y_batches) + len(already_seen_y_set) == len(self.dataset) / num_replicas
            assert len(left_y_batches | already_seen_y_set) == len(self.dataset) / num_replicas
            dist.barrier()
        finally:
            rank_zero_rm(path)

        # if dist.is_initialized():
        #     dist.destroy_process_group()

    @magic_argv_env_context
    @pytest.mark.parametrize("only_state_dict", ([True, False]))
    @pytest.mark.parametrize("fp16", ([True, False]))
    @pytest.mark.parametrize("device", ([[0,1]]))
    def test_save_and_load_with_randomsampler(self, device, only_state_dict, fp16):
        """
        测试save和load函数，主要测试 dataloader 被替换了 batch_sampler 的情况
        """

        try:
            path = "checkpoints/"

            num_replicas = len(device)

            dataloader = dataloader_with_randomsampler(self.dataset, 4, True, False, unrepeated=False)
            dataloader.batch_sampler.sampler.set_distributed(
                num_replicas=int(os.getenv("WORLD_SIZE", "1")),
                rank=int(os.getenv("RANK", "0")),
                pad=True
            )
            num_consumed_batches = 4
            
            driver1 = generate_driver(20, 1, device=device, fp16=fp16, train_dataloader=dataloader)
            driver2 = generate_driver(20, 1, device=device, fp16=False, train_dataloader=dataloader)

            already_seen_x_set = set()
            already_seen_y_set = set()
            driver1.set_sampler_epoch(dataloader, 4)
            for idx, batch in enumerate(dataloader):
                if idx >= num_consumed_batches:
                    break
                already_seen_x_set.update(batch["x"].reshape(-1, ).tolist())
                already_seen_y_set.update(batch["y"].reshape(-1, ).tolist())

            # 同步
            dist.barrier()

            # 保存状态
            sampler_states = dataloader.batch_sampler.sampler.state_dict()
            save_states = {"num_consumed_batches": num_consumed_batches}
            if only_state_dict:
                driver1.save_checkpoint(Path(path), save_states, dataloader, only_state_dict, should_save_model=True)
            else:
                driver1.save_checkpoint(Path(path), save_states, dataloader, only_state_dict, should_save_model=True, input_spec=[torch.ones((16, 10))])
            dist.barrier()  # 等待save成功
            # 加载
            # 更改 batch_size
            dataloader = dataloader_with_randomsampler(self.dataset, 2, True, False, unrepeated=False)
            dataloader.batch_sampler.sampler.set_distributed(
                num_replicas=driver2.world_size,
                rank=driver2.global_rank,
                pad=True
            )
            load_states = driver2.load_checkpoint(Path(path), dataloader, only_state_dict, should_load_model=True)
            replaced_loader = load_states.pop("dataloader")

            # 1. 检查 optimizer 的状态
            # TODO optimizer 的 state_dict 总是为空

            # 2. 检查 sampler 是否被正确地加载和替换
            assert not (replaced_loader is dataloader)
            assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
            if os.environ['FASTNLP_GLOBAL_RANK'] == '0':
                assert replaced_loader.batch_sampler.sampler.seed == sampler_states["seed"]
                assert replaced_loader.batch_sampler.sampler.epoch == sampler_states["epoch"]
                assert len(replaced_loader.batch_sampler.sampler.dataset) == sampler_states["length"]
                assert replaced_loader.batch_sampler.sampler.shuffle == sampler_states["shuffle"]
            assert replaced_loader.batch_sampler.sampler.num_consumed_samples == 4 * num_consumed_batches * num_replicas

            # 3. 检查 fp16 是否被加载
            if fp16:
                assert not isinstance(driver2.grad_scaler, torch.cuda.amp.GradScaler)

            # 4. 检查 model 的参数是否正确
            # 5. 检查 batch_idx
            start_batch = load_states.pop('batch_idx_in_epoch')
            assert start_batch == 2 * num_consumed_batches
            left_x_batches = set()
            left_y_batches = set()
            driver2.set_sampler_epoch(replaced_loader, 4)
            for idx, batch in enumerate(replaced_loader):

                left_x_batches.update(batch["x"].reshape(-1, ).tolist())
                left_y_batches.update(batch["y"].reshape(-1, ).tolist())
                batch = driver1.move_data_to_device(batch)
                res1 = driver1.model(
                    batch,
                    fastnlp_fn=driver1.model.module.model.evaluate_step,
                    # Driver.model -> DataParallel.module -> _FleetWrappingModel.model
                    fastnlp_signature_fn=None,
                    wo_auto_param_call=False,
                )
                res2 = driver2.model(
                    batch,
                    fastnlp_fn=driver2.model.module.model.evaluate_step,
                    fastnlp_signature_fn=None,
                    wo_auto_param_call=False,
                )
                assert torch.equal(res1["preds"], res2["preds"])

            assert len(left_x_batches) + len(already_seen_x_set) == len(self.dataset) / num_replicas
            assert len(left_x_batches | already_seen_x_set) == len(self.dataset) / num_replicas
            assert len(left_y_batches) + len(already_seen_y_set) == len(self.dataset) / num_replicas
            assert len(left_y_batches | already_seen_y_set) == len(self.dataset) / num_replicas

        finally:
            rank_zero_rm(path)

        # if dist.is_initialized():
        #     dist.destroy_process_group()
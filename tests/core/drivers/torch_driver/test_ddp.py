import os

import pytest
from pathlib import Path

from fastNLP.core.drivers.torch_driver.ddp import TorchDDPDriver
from fastNLP import prepare_torch_dataloader
from fastNLP.core.samplers import (
    RandomSampler,
    UnrepeatedSampler,
    BucketedBatchSampler,
    UnrepeatedRandomSampler,
    UnrepeatedSequentialSampler,
)
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.datasets.torch_data import TorchNormalDataset, TorchNormalXYDataset
from tests.helpers.utils import magic_argv_env_context, recover_logger, Capturing
from fastNLP.envs.distributed import rank_zero_rm
from fastNLP import logger
from fastNLP.core.drivers.torch_driver.dist_utils import fastnlp_torch_all_gather
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, BatchSampler

def generate_driver(labels, features, device=[0,1], fp16=False, output_from_new_proc="all"):
    torch_model = TorchNormalModel_Classification_1(labels, features)
    torch_opt = torch.optim.Adam(params=torch_model.parameters(), lr=0.01)
    device = [torch.device(i) for i in device]
    driver = TorchDDPDriver(
        model=torch_model,
        parallel_device=device,
        fp16=fp16,
        output_from_new_proc=output_from_new_proc
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
# 测试 TorchDDPDriver 的一些函数
#
############################################################################

@pytest.mark.torch
@magic_argv_env_context
def test_multi_drivers():
    """
    测试使用了多个 TorchDDPDriver 的情况。
    """
    generate_driver(10, 10)
    generate_driver(20, 10)
    
    with pytest.raises(RuntimeError):
        # 设备设置不同，应该报错
        generate_driver(20, 3, device=[0,1,2])
        assert False
    dist.barrier()

    if dist.is_initialized():
        dist.destroy_process_group()

@pytest.mark.torch
class TestDDPDriverFunction:
    """
    测试 TorchDDPDriver 一些简单函数的测试类，基本都是测试能否运行、是否存在 import 错误等问题
    """

    @magic_argv_env_context
    def test_simple_functions(self):
        """
        简单测试多个函数
        """
        driver = generate_driver(10, 10)

        """
        测试 move_data_to_device 函数。这个函数仅调用了 torch_move_data_to_device ，测试例在
        tests/core/utils/test_torch_utils.py中，就不重复测试了
        """
        driver.move_data_to_device(torch.rand((32, 64)))
        dist.barrier()

        """
        测试 is_distributed 函数
        """
        assert driver.is_distributed() == True
        dist.barrier()

        """
        测试 get_no_sync_context 函数
        """
        res = driver.get_model_no_sync_context()
        dist.barrier()

        """
        测试 is_global_zero 函数
        """
        driver.is_global_zero()
        dist.barrier()

        """
        测试 unwrap_model 函数
        """
        driver.unwrap_model()
        dist.barrier()

        """
        测试 get_local_rank 函数
        """
        driver.get_local_rank()
        dist.barrier()

        """
        测试 all_gather 函数
        详细的测试在 test_dist_utils.py 中完成
        """
        obj = {
            "rank": driver.global_rank
        }
        obj_list = driver.all_gather(obj, group=None)
        for i, res in enumerate(obj_list):
            assert res["rank"] == i

        """
        测试 broadcast_object 函数
        详细的函数在 test_dist_utils.py 中完成
        """
        if driver.global_rank == 0:
            obj = {
                "rank": driver.global_rank
            }
        else:
            obj = None
        res = driver.broadcast_object(obj, src=0)
        assert res["rank"] == 0

        if dist.is_initialized():
            dist.destroy_process_group()

############################################################################
#
# 测试 set_dist_repro_dataloader 函数
#
############################################################################

@pytest.mark.torch
class TestSetDistReproDataloader:

    @classmethod
    def setup_class(cls):
        cls.device = [0, 1]

    def setup_method(self):
        self.dataset = TorchNormalDataset(100)

    """
    传入的 `dist` 参数为具体的 ReproducibleSampler 或 ReproducibleBatchSampler 的情况
    此时对应 driver.load_checkpoint 中的情况
    """

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_batch_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 BucketedBatchSampler 时的表现
        此时应该将 batch_sampler 替换为 dist 对应的 BucketedBatchSampler
        """
        driver = generate_driver(10, 10, device=self.device)
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=not shuffle)
        batch_sampler = BucketedBatchSampler(self.dataset, self.dataset._data, batch_size=4, shuffle=shuffle)
        replaced_loader = driver.set_dist_repro_dataloader(dataloader, batch_sampler, False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BucketedBatchSampler)
        assert replaced_loader.batch_sampler is batch_sampler
        self.check_distributed_sampler(replaced_loader.batch_sampler)
        self.check_set_dist_repro_dataloader(driver, dataloader, replaced_loader, shuffle)
        
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 RandomSampler 时的表现
        此时应该将 batch_sampler.sampler 替换为 dist 对应的 RandomSampler
        """
        driver = generate_driver(10, 10, device=self.device)
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=not shuffle)
        sampler = RandomSampler(self.dataset, shuffle=shuffle)
        replaced_loader = driver.set_dist_repro_dataloader(dataloader, sampler, False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BatchSampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert replaced_loader.batch_sampler.sampler is sampler
        assert replaced_loader.batch_sampler.batch_size == dataloader.batch_sampler.batch_size
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        self.check_set_dist_repro_dataloader(driver, dataloader, replaced_loader, shuffle)

        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()
    
    """
    传入的参数 `dist` 为 None 的情况，这种情况出现在 trainer 和 evaluator 的初始化过程中，用户指定了 `use_dist_sampler` 
    参数为 False。此时函数会根据 `reproducible` 的设置进行不同的处理。
    当 `reproducible` 为 False 时，需要根据 dataloader 的 batch_sampler 或 sampler 是否为 Reproducible 来决定
    是否重新实例化 dataloader
    """

    @magic_argv_env_context
    def test_with_dist_none_reproducible_true(self):
        """
        测试 set_dist_repro_dataloader 中 dist 为 None、reproducible 为 True 时的表现
        当用户在 driver 之外初始化了分布式环境时，fastnlp 不支持进行断点重训，此时应该报错
        """
        driver = generate_driver(10, 10, device=self.device)
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=True)
        with pytest.raises(RuntimeError):
            # 应当抛出 RuntimeError
            replaced_loader = driver.set_dist_repro_dataloader(dataloader, None, True)

        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_none_reproducible_false_dataloader_reproducible_batch_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 None、reproducible 为 False 、dataloader 有 BucketedBatchSampler 
        时的表现
        此时传入的 dataloader 的 batch_sampler 应该已经执行了 set_distributed，产生一个新的 dataloader，其 batch_sampler
        和原 dataloader 相同
        """
        driver = generate_driver(10, 10, device=self.device)
        dataloader = dataloader_with_bucketedbatchsampler(self.dataset, self.dataset._data, 4, shuffle, False)
        dataloader.batch_sampler.set_distributed(
            num_replicas=driver.world_size,
            rank=driver.global_rank,
            pad=True
        )
        replaced_loader = driver.set_dist_repro_dataloader(dataloader, None, False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BucketedBatchSampler)
        assert replaced_loader.batch_sampler.batch_size == 4
        self.check_distributed_sampler(dataloader.batch_sampler)
        self.check_set_dist_repro_dataloader(driver, dataloader, replaced_loader, shuffle)

        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_none_reproducible_false_dataloader_reproducible_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 None、reproducible 为 False 、dataloader 有 RandomSampler 时的表现
        此时传入的 dataloader 的 batch_sampler.sampler 应该已经执行了 set_distributed，产生一个新的 dataloader，其
        batch_sampler.sampler 和原 dataloader 相同
        """
        driver = generate_driver(10, 10, device=self.device)
        dataloader = dataloader_with_randomsampler(self.dataset, 4, shuffle, False, unrepeated=False)
        dataloader.batch_sampler.sampler.set_distributed(
            num_replicas=driver.world_size,
            rank=driver.global_rank
        )
        replaced_loader = driver.set_dist_repro_dataloader(dataloader, None, False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
        assert not (replaced_loader.batch_sampler.sampler is dataloader.batch_sampler.sampler)
        assert replaced_loader.batch_sampler.batch_size == 4
        assert replaced_loader.batch_sampler.drop_last == False
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        self.check_set_dist_repro_dataloader(driver, dataloader, replaced_loader, shuffle)
    
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_none_reproducible_false_dataloader_normal(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 None、reproducible 为 False 、dataloader 为一般情况时的表现
        此时直接返回原来的 dataloader，不做任何处理。
        """
        driver = generate_driver(10, 10, device=self.device)
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=shuffle)
        replaced_loader = driver.set_dist_repro_dataloader(dataloader, None, False)

        assert replaced_loader is dataloader
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()

    """
    传入的参数 `dist` 为 'dist' 的情况，这种情况出现在 trainer 的初始化过程中，用户指定了 `use_dist_sampler` 参数
    为 True。此时函数会根据 dataloader 的 batch_sampler 或 sampler 是否为 Reproducible 来决定如何重新实例化 dataloader
    """

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_dist_dataloader_reproducible_batch_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'dist'、dataloader.batch_sampler 为 ReproducibleBatchSampler
        的表现
        此时应该返回一个新的 dataloader，其batch_sampler 和原 dataloader 相同，且应该正确地设置了分布式相关的属性
        """
        driver = generate_driver(10, 10, device=self.device)
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_sampler=BucketedBatchSampler(self.dataset, self.dataset._data, batch_size=4, shuffle=shuffle)
        )
        dataloader = dataloader_with_bucketedbatchsampler(self.dataset, self.dataset._data, 4, shuffle, False)
        replaced_loader = driver.set_dist_repro_dataloader(dataloader, "dist", False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BucketedBatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert replaced_loader.batch_sampler.batch_size == 4
        assert replaced_loader.drop_last == dataloader.drop_last
        self.check_distributed_sampler(replaced_loader.batch_sampler)
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_dist_dataloader_reproducible_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'dist'、dataloader.batch_sampler.sampler 为 ReproducibleSampler
        的表现
        此时应该返回一个新的 dataloader，其 batch_sampler.sampler 和原 dataloader 相同，且应该正确地设置了分布式相关
        的属性
        """
        driver = generate_driver(10, 10, device=self.device)
        dataloader = dataloader_with_randomsampler(self.dataset, 4, shuffle, False, unrepeated=False)
        replaced_loader = driver.set_dist_repro_dataloader(dataloader, "dist", False)

        assert not (replaced_loader is dataloader)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
        assert not (replaced_loader.batch_sampler.sampler is dataloader.batch_sampler.sampler)
        assert replaced_loader.batch_sampler.batch_size == 4
        assert replaced_loader.batch_sampler.sampler.shuffle == shuffle
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_dist_dataloader_normal(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'dist'、dataloader 为一般情况的表现
        此时应该返回一个新的 dataloader，并替换其 batch_sampler.sampler 为 RandomSampler，且应该正确设置了分布式相关
        的属性
        """
        driver = generate_driver(10, 10, device=self.device)
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=shuffle)
        replaced_loader = driver.set_dist_repro_dataloader(dataloader, "dist", False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
        assert replaced_loader.batch_sampler.batch_size == dataloader.batch_sampler.batch_size
        assert replaced_loader.batch_sampler.sampler.shuffle == shuffle
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()

    """
    传入的参数 `dist` 为 'unrepeatdist' 的情况，这种情况出现在 evaluator 的初始化过程中，用户指定了 `use_dist_sampler` 参数
    为 True。此时函数会根据 dataloader 的  sampler 是否为 Unrepeated 和 Reproducible 来决定如何重新实例化 dataloader
    """

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_unrepeat_dataloader_reproducible_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'unrepeatdist'、dataloader.batch_sampler.sampler 为 ReproducibleSampler
        的表现
        此时应该返回一个新的 dataloader，且将原来的 Sampler 替换为 UnrepeatedRandomSampler，且正确地设置了分布式相关
        的属性
        """
        driver = generate_driver(10, 10, device=self.device)
        dataloader = dataloader_with_randomsampler(self.dataset, 4, shuffle, False, unrepeated=False)
        replaced_loader = driver.set_dist_repro_dataloader(dataloader, "unrepeatdist", False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, UnrepeatedRandomSampler)
        assert replaced_loader.batch_sampler.batch_size == 4
        assert replaced_loader.batch_sampler.sampler.shuffle == shuffle
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_unrepeat_dataloader_unrepreated_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'unrepeatdist'、dataloader.batch_sampler.sampler 为 UnrepeatedSampler
        的表现
        此时应该返回一个新的 dataloader，且重新实例化了原来的 Sampler
        """
        driver = generate_driver(10, 10, device=self.device)
        dataloader = dataloader_with_randomsampler(self.dataset, 4, shuffle, False, unrepeated=True)
        replaced_loader = driver.set_dist_repro_dataloader(dataloader, "unrepeatdist", False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, UnrepeatedRandomSampler)
        assert not (replaced_loader.batch_sampler.sampler is dataloader.batch_sampler.sampler)
        assert replaced_loader.batch_sampler.batch_size == 4
        assert replaced_loader.drop_last == dataloader.drop_last
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_unrepeat_dataloader_normal(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'unrepeatdist'、dataloader 为一般情况的表现
        此时应该返回一个新的 dataloader，且将 sampler 替换为 UnrepeatedSequentialSampler，并正确地设置了分布式相关
        的属性
        """
        driver = generate_driver(10, 10, device=self.device)
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=shuffle)
        replaced_loader = driver.set_dist_repro_dataloader(dataloader, "unrepeatdist", False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, UnrepeatedSequentialSampler)
        assert replaced_loader.batch_sampler.batch_size == 4
        assert replaced_loader.drop_last == dataloader.drop_last
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        dist.barrier()
        if dist.is_initialized():
            dist.destroy_process_group()

    def check_distributed_sampler(self, sampler):
        """
        测试替换得到的 sampler 或 batch_sampler 的分布式设置是否正确
        """
        assert sampler.num_replicas == dist.get_world_size()
        assert sampler.rank == dist.get_rank()
        if not isinstance(sampler, UnrepeatedSampler):
            assert sampler.pad == True

    def check_set_dist_repro_dataloader(self, driver, dataloader, replaced_loader, shuffle):
        """
        测试多卡下 set_dist_repro_dataloader 函数的执行结果是否正确
        """
        # 迭代两个 batch
        num_replicas = len(self.device)
        num_consumed_batches = 2
        already_seen_idx = set()
        if isinstance(replaced_loader.batch_sampler, BucketedBatchSampler):
            sampler_states = replaced_loader.batch_sampler.set_epoch(4)
        else:
            sampler_states = replaced_loader.batch_sampler.sampler.set_epoch(4)
        for idx, batch in enumerate(replaced_loader):
            if idx >= num_consumed_batches:
                break
            already_seen_idx.update(batch.tolist())
        dist.barrier()
        if isinstance(replaced_loader.batch_sampler, BucketedBatchSampler):
            sampler_states = replaced_loader.batch_sampler.state_dict()
        else:
            sampler_states = replaced_loader.batch_sampler.sampler.state_dict()

        # 重新加载，应该可以输出剩下的内容，且对于 TorchNormalDataset 来说，排序后应该是一个 range
        left_idxes = set()
        if isinstance(replaced_loader.batch_sampler, BucketedBatchSampler):
            batch_size = replaced_loader.batch_sampler.batch_size
            sampler_states["num_consumed_samples"] = num_consumed_batches * batch_size * num_replicas
            # 重新改造 dataloader
            new_loader = dataloader_with_bucketedbatchsampler(
                replaced_loader.dataset,
                length=replaced_loader.dataset._data,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=False,
            )
            new_loader.batch_sampler.set_distributed(
                num_replicas=driver.world_size,
                rank=driver.global_rank,
                pad=True
            )
            new_loader.batch_sampler.load_state_dict(sampler_states)
            new_loader.batch_sampler.set_epoch(4)
        else:
            batch_size = replaced_loader.batch_sampler.batch_size
            sampler_states["num_consumed_samples"] = num_consumed_batches * batch_size * num_replicas
            # 重新构造 dataloader
            new_loader = dataloader_with_randomsampler(replaced_loader.dataset, batch_size, shuffle, drop_last=False)
            new_loader.batch_sampler.sampler.set_distributed(
                num_replicas=driver.world_size,
                rank=driver.global_rank
            )
            new_loader.batch_sampler.sampler.load_state_dict(sampler_states)
            new_loader.batch_sampler.sampler.set_epoch(4)
        for idx, batch in enumerate(new_loader):
            left_idxes.update(batch.tolist())

        assert len(left_idxes) + len(already_seen_idx) == len(self.dataset) / num_replicas
        assert len(left_idxes | already_seen_idx) == len(self.dataset) / num_replicas


############################################################################
#
# 测试 save 和 load 相关的功能
#
############################################################################
@pytest.mark.torch
class TestSaveLoad:
    """
    测试多卡情况下 save 和 load 相关函数的表现
    """

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
            driver1, driver2 = generate_driver(20, 1), generate_driver(20, 1)

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

        if dist.is_initialized():
            dist.destroy_process_group()

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

            driver1, driver2 = generate_driver(20, 1, device=device, fp16=fp16), \
                                            generate_driver(20, 1, device=device, fp16=False)
            dataloader = dataloader_with_bucketedbatchsampler(
                self.dataset,
                length=[10 for i in range(len(self.dataset))],
                batch_size=4,
                shuffle=True,
                drop_last=False
            )
            dataloader.batch_sampler.set_distributed(
                num_replicas=driver1.world_size,
                rank=driver1.global_rank,
                pad=True
            )
            num_consumed_batches = 4

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

        if dist.is_initialized():
            dist.destroy_process_group()

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

            driver1 = generate_driver(20, 1, device=device, fp16=fp16)
            driver2 = generate_driver(20, 1, device=device, fp16=False)

            dataloader = dataloader_with_randomsampler(self.dataset, 4, True, False, unrepeated=False)
            dataloader.batch_sampler.sampler.set_distributed(
                num_replicas=driver1.world_size,
                rank=driver1.global_rank,
                pad=True
            )
            num_consumed_batches = 4

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
            driver1.save_checkpoint(Path(path), save_states, dataloader, only_state_dict, should_save_model=True)
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

        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.torch
@magic_argv_env_context
@pytest.mark.parametrize("shuffle", ([True, False]))
@pytest.mark.parametrize("batch_size", ([1, 3, 16, 17]))
@pytest.mark.parametrize("drop_last", ([True, False]))
def test_shuffle_dataloader(shuffle, batch_size, drop_last, reproducible=True):
    try:
        # 需要检验一下 set_dist_repro_dataloader 没有修改参数
        num_samples = 200
        dataset = TorchNormalXYDataset(num_samples)
        dl = prepare_torch_dataloader(dataset, shuffle=shuffle, batch_size=batch_size, drop_last=drop_last)
        model = TorchNormalModel_Classification_1(10, 32)
        device = [torch.device(i) for i in [0, 1]]

        driver = TorchDDPDriver(model, parallel_device=device)
        driver.setup()
        dl = driver.set_dist_repro_dataloader(dataloader=dl, dist='dist', reproducible=reproducible)

        data = []
        flags = []
        for batch in dl:
            flags.append(batch['x'].size(0) == batch_size)
            data.extend(batch['x'].reshape(-1).tolist())

        _num_samples = num_samples//2

        if drop_last and _num_samples%batch_size != 0:
            assert len(data)!=_num_samples
            assert all(flags) == True
        elif _num_samples%batch_size!=0:
            assert flags[-1] is False
        else:
            assert len(data) == _num_samples

        if not shuffle:
            for i in range(1, len(data)-1):
                assert data[i]>data[i-1]
        else:
            flags = []
            for i in range(1, len(data)-1):
                flags.append(data[i]>data[i-1])
            assert all(flags) is False
        datas = fastnlp_torch_all_gather(data)
        if drop_last:
            assert len(set(datas[0] + datas[1])) == num_samples-_num_samples%batch_size*2
        else:
            assert len(set(datas[0] + datas[1])) == num_samples
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


@pytest.mark.torch
@magic_argv_env_context
@pytest.mark.parametrize("shuffle", ([True, False]))
@pytest.mark.parametrize("batch_size", ([1, 3, 16, 17]))
@pytest.mark.parametrize("drop_last", ([True, False]))
def test_batch_sampler_dataloader(shuffle, batch_size, drop_last, reproducible=True):
    try:
        # 需要检验一下 set_dist_repro_dataloader 没有修改参数
        num_samples = 200
        num_device = 2
        dataset = TorchNormalXYDataset(num_samples)
        sampler = BucketedBatchSampler(dataset, length=dataset._data, batch_size=batch_size, drop_last=drop_last,
                                       shuffle=shuffle, num_batch_per_bucket=2)
        dl = prepare_torch_dataloader(dataset, batch_sampler=sampler)
        model = TorchNormalModel_Classification_1(10, 32)
        device = [torch.device(i) for i in [0, 1]]
        driver = TorchDDPDriver(model, parallel_device=device)
        driver.setup()
        dl = driver.set_dist_repro_dataloader(dataloader=dl, dist='dist', reproducible=reproducible)

        data = []
        flags = []
        for batch in dl:
            d = batch['x'].reshape(-1).tolist()
            diff = max(d) - min(d)
            assert diff<batch_size*2*2*2
            data.extend(d)
            flags.append(len(d)==batch_size)
        _num_samples = num_samples//num_device
        if drop_last and _num_samples%batch_size != 0:
            assert len(data)!=num_samples
            assert all(flags) == True
        elif _num_samples%batch_size!=0:
            assert flags[-1] is False
        else:
            assert len(data) == _num_samples

        if not shuffle:
            for i in range(1, len(data)-1):
                assert data[i]<data[i-1]
        else:
            flags = []
            for i in range(1, len(data)-1):
                flags.append(data[i]<data[i-1])
            assert all(flags) is False
        if dist.is_initialized():
            datas = fastnlp_torch_all_gather(data)
            if drop_last:
                assert len(set(datas[0] + datas[1])) == num_samples-_num_samples%batch_size*2
            else:
                assert len(set(datas[0] + datas[1])) == num_samples
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()



@pytest.mark.torch
@magic_argv_env_context
@recover_logger
@pytest.mark.parametrize("inherit", ([True, False]))
def test_customized_batch_sampler_dataloader(inherit):
    try:
        logger.set_stdout('raw', level='info')
        # 需要检验一下 set_dist_repro_dataloader 是否可以在定制 batch_sampler 的情况下正确运行
        num_samples = 10
        dataset = TorchNormalXYDataset(num_samples)
        if inherit:
            class BatchSampler(torch.utils.data.BatchSampler):
                def __init__(self, dataset, batch_size):
                    self.dataset = dataset
                    self.batch_size = batch_size

                def __iter__(self):
                    indices = list(range(len(dataset)))
                    for i in range(len(self)):
                        start = i * self.batch_size
                        end = (i + 1) * self.batch_size
                        return indices[start:end]

                def __len__(self):
                    return (len(self.dataset)+self.batch_size-1)//self.batch_size
        else:
            class BatchSampler:
                def __init__(self, dataset, batch_size):
                    self.dataset = dataset
                    self.batch_size = batch_size

                def __iter__(self):
                    indices = list(range(len(dataset)))
                    for i in range(len(self)):
                        start = i * self.batch_size
                        end = (i + 1) * self.batch_size
                        return indices[start:end]

                def __len__(self):
                    return (len(self.dataset)+self.batch_size-1)//self.batch_size

        dl = prepare_torch_dataloader(dataset, batch_sampler=BatchSampler(dataset, batch_size=4))
        model = TorchNormalModel_Classification_1(10, 32)
        device = [torch.device(i) for i in [0, 1]]
        driver = TorchDDPDriver(model, parallel_device=device)
        driver.setup()
        # TODO 这里需要raise
        with pytest.raises(TypeError):
            dl = driver.set_dist_repro_dataloader(dataloader=dl, dist='dist', reproducible=False)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


@pytest.mark.torch
@magic_argv_env_context
@recover_logger
@pytest.mark.parametrize("inherit", ([True, False]))
def test_customized_sampler_dataloader(inherit):
    try:
        logger.set_stdout('raw', level='info')
        # 需要检验一下 set_dist_repro_dataloader 是否可以在定制 batch_sampler 的情况下正确运行
        num_samples = 10
        dataset = TorchNormalXYDataset(num_samples)
        if inherit:
            class Sampler(torch.utils.data.RandomSampler):
                def __init__(self, dataset, batch_size):
                    self.dataset = dataset
                    self.batch_size = batch_size

                def __iter__(self):
                    indices = list(range(len(dataset)))
                    return iter(indices)

                def __len__(self):
                    return len(self.dataset)
        else:
            class Sampler:
                def __init__(self, dataset, batch_size):
                    self.dataset = dataset
                    self.batch_size = batch_size

                def __iter__(self):
                    indices = list(range(len(dataset)))
                    return iter(indices)

                def __len__(self):
                    return len(self.dataset)

        dl = prepare_torch_dataloader(dataset, sampler=Sampler(dataset, batch_size=4))
        model = TorchNormalModel_Classification_1(10, 32)
        device = [torch.device(i) for i in [0, 1]]
        driver = TorchDDPDriver(model, parallel_device=device)
        driver.setup()
        # TODO 这里需要raise
        with pytest.raises(TypeError):
            dl = driver.set_dist_repro_dataloader(dataloader=dl, dist='dist', reproducible=False)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


@pytest.mark.torch
@magic_argv_env_context
@pytest.mark.parametrize("device", (['cpu', 0, [0, 1]]))
def test_sync_batchnorm(device):
    import numpy as np
    from fastNLP import Event, Trainer, DataSet, Instance
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dense1 = torch.nn.Linear(in_features=1, out_features=10)
            self.norm = torch.nn.BatchNorm1d(10, affine=False)
            self.dense2 = torch.nn.Linear(in_features=10, out_features=1)

        def forward(self, x, y):
            x = self.dense1(x)
            x = self.norm(x)
            x = self.dense2(x)
            loss = torch.nn.functional.mse_loss(x, y)
            return dict(loss=loss)

    @Trainer.on(Event.on_train_batch_end())
    def check_sync(trainer):
        if not isinstance(device, list) or len(device) is 0:
            # 单卡或 CPU 的情况下不需要检查
            return
        running_mean_list = fastnlp_torch_all_gather(trainer.model.norm.running_mean)
        running_var_list = fastnlp_torch_all_gather(trainer.model.norm.running_var)
        for running_mean in running_mean_list:
            # 检查每张卡上的均值是否相等
            assert running_mean_list[0].equal(running_mean)
        for running_var in running_var_list:
            # 检查每张卡上的方差是否相等
            assert running_var_list[0].equal(running_var)

    model = Model()
    dataset = DataSet()
    for i in np.arange(30):
        dataset.append(Instance(x=np.random.rand(10, 1), y=np.random.rand(10, 1)))
    dl = prepare_torch_dataloader(dataset, batch_size=10)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model,
        train_dataloader=dl,
        device=device,
        optimizers=optimizer,
        sync_bn=True,
        n_epochs=1
    )
    trainer.run()
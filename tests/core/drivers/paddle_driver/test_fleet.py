from dataclasses import replace
import pytest
import os

os.environ["FASTNLP_BACKEND"] = "paddle"
from fastNLP.core.drivers.paddle_driver.fleet import PaddleFleetDriver
from fastNLP.core.samplers import (
    RandomSampler,
    UnrepeatedSampler,
    BucketedBatchSampler,
    UnrepeatedRandomSampler,
    UnrepeatedSequentialSampler,
)
from tests.helpers.models.paddle_model import PaddleNormalModel_Classification_1
from tests.helpers.datasets.paddle_data import PaddleNormalDataset
from tests.helpers.utils import magic_argv_env_context

import paddle
import paddle.distributed as dist
from paddle.io import DataLoader, BatchSampler

def generate_driver(num_labels, feature_dimension):
    paddle_model = PaddleNormalModel_Classification_1(num_labels, feature_dimension)
    paddle_opt = paddle.optimizer.Adam(parameters=paddle_model.parameters(), learning_rate=0.01)
    driver = PaddleFleetDriver(
        model=paddle_model,
        parallel_device=[0,1],
    )
    driver.set_optimizers(paddle_opt)
    driver.setup()

    return driver

############################################################################
#
# 测试PaddleFleetDriver的一些函数
#
############################################################################

class TestFleetDriverFunction:
    """
    测试 PaddleFleetDriver 一些简单函数的测试类，基本都是测试能否运行、是否存在 import 错误等问题
    """

    @classmethod
    def setup_class(cls):
        cls.driver = generate_driver(10, 10)

    @magic_argv_env_context
    def test_move_data_to_device(self):
        """
        这个函数仅调用了paddle_move_data_to_device，测试例在tests/core/utils/test_paddle_utils.py中
        就不重复测试了
        """
        self.driver.move_data_to_device(paddle.rand((32, 64)))

        dist.barrier()

    @magic_argv_env_context
    def test_is_distributed(self):
        """
        测试 is_distributed 函数
        """
        assert self.driver.is_distributed() == True
        dist.barrier()

    @magic_argv_env_context
    def test_get_no_sync_context(self):
        """
        测试 get_no_sync_context 函数
        """
        res = self.driver.get_no_sync_context()
        dist.barrier()

    @magic_argv_env_context
    def test_is_global_zero(self):
        """
        测试 is_global_zero 函数
        """
        self.driver.is_global_zero()
        dist.barrier()

    @magic_argv_env_context
    def test_unwrap_model(self):
        """
        测试 unwrap_model 函数
        """
        self.driver.unwrap_model()
        dist.barrier()

    @magic_argv_env_context
    def test_get_local_rank(self):
        """
        测试 get_local_rank 函数
        """
        self.driver.get_local_rank()
        dist.barrier()

############################################################################
#
# 测试 set_dist_repro_dataloader 函数
#
############################################################################

class TestSetDistReproDataloader:

    @classmethod
    def setup_class(cls):
        cls.driver = generate_driver(10, 10)

    def setup_method(self):
        self.dataset = PaddleNormalDataset(20)

    """
    传入的 `dist` 参数为具体的 ReproducibleSampler 或 ReproducibleBatchSampler 的情况
    此时对应 driver.load 中的情况
    """

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_set_dist_repro_dataloader_with_dist_batch_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 BucketedBatchSampler 时的表现
        """
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=not shuffle)
        batch_sampler = BucketedBatchSampler(self.dataset, self.dataset._data, batch_size=4, shuffle=shuffle)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, batch_sampler, False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BucketedBatchSampler)
        assert replaced_loader.batch_sampler is batch_sampler
        self.check_distributed_sampler(replaced_loader.batch_sampler)
        
        dist.barrier()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_set_dist_repro_dataloader_with_dist_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 RandomSampler 时的表现
        """
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=not shuffle)
        sampler = RandomSampler(self.dataset, shuffle=shuffle)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, sampler, False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BatchSampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert replaced_loader.batch_sampler.sampler is sampler
        assert replaced_loader.batch_sampler.batch_size == dataloader.batch_sampler.batch_size
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)

        dist.barrier()
    
    """
    传入的参数 `dist` 为 None 的情况，这种情况出现在 trainer 和 evaluator 的初始化过程中，用户指定了 `use_dist_sampler` 
    参数为 False。此时函数会根据 `reproducible` 的设置进行不同的处理。
    当 `reproducible` 为 False 时，需要根据 dataloader 的 batch_sampler 或 sampler 是否为 Reproducible 来决定
    是否重新实例化 dataloader
    """

    @magic_argv_env_context
    def test_set_dist_repro_dataloader_with_dist_none_reproducible_true(self):
        """
        测试 set_dist_repro_dataloader 中 dist 为 None、reproducible 为 True 时的表现
        """
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=True)
        with pytest.raises(RuntimeError):
            # 应当抛出 RuntimeError
            replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, None, True)

        dist.barrier()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_set_dist_repro_dataloader_with_dist_none_reproducible_false_dataloader_reproducible_batch_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 None、reproducible 为 False 、dataloader 有 BucketedBatchSampler 
        时的表现
        """
        dataloader = DataLoader(
            self.dataset,
            batch_sampler = BucketedBatchSampler(self.dataset, self.dataset._data, batch_size=4, shuffle=shuffle),
        )
        dataloader.batch_sampler.set_distributed(
            num_replicas=self.driver.world_size,
            rank=self.driver.global_rank,
            pad=True
        )
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, None, False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BucketedBatchSampler)
        assert replaced_loader.batch_sampler.batch_size == 4
        self.check_distributed_sampler(dataloader.batch_sampler)

        dist.barrier()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_set_dist_repro_dataloader_with_dist_none_reproducible_false_dataloader_reproducible_smpler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 None、reproducible 为 False 、dataloader 有 RandomSampler 时的表现
        """
        batch_sampler = BatchSampler(dataset=self.dataset, batch_size=2)
        batch_sampler.sampler = RandomSampler(self.dataset, shuffle)
        batch_sampler.sampler.set_distributed(
            num_replicas=self.driver.world_size,
            rank=self.driver.global_rank
        )
        dataloader = DataLoader(
            self.dataset,
            batch_sampler=batch_sampler
        )
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, None, False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
        assert not (replaced_loader.batch_sampler.sampler is dataloader.batch_sampler.sampler)
        assert replaced_loader.batch_sampler.batch_size == 2
        assert replaced_loader.batch_sampler.drop_last == False
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        dist.barrier()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_set_dist_repro_dataloader_with_dist_none_reproducible_false_dataloader_normal(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 None、reproducible 为 False 、dataloader 为一般情况时的表现
        """
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=shuffle)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, None, False)

        assert replaced_loader is dataloader
        dist.barrier()

    """
    传入的参数 `dist` 为 'dist' 的情况，这种情况出现在 trainer 的初始化过程中，用户指定了 `use_dist_sampler` 参数
    为 True。此时函数会根据 dataloader 的 batch_sampler 或 sampler 是否为 Reproducible 来决定如何重新实例化 dataloader
    """

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_set_dist_repro_dataloader_with_dist_dist_dataloader_reproducible_batch_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'dist'、dataloader.batch_sampler 为 ReproducibleBatchSampler
        的表现
        """
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_sampler=BucketedBatchSampler(self.dataset, self.dataset._data, batch_size=4, shuffle=shuffle)
        )
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, "dist", False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BucketedBatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert replaced_loader.batch_sampler.batch_size == 4
        assert replaced_loader.drop_last == dataloader.drop_last
        self.check_distributed_sampler(replaced_loader.batch_sampler)
        dist.barrier()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_set_dist_repro_dataloader_with_dist_dist_dataloader_reproducible_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'dist'、dataloader.batch_sampler.sampler 为 ReproducibleSampler
        的表现
        """
        batch_sampler = BatchSampler(dataset=self.dataset, batch_size=2, shuffle=shuffle)
        batch_sampler.sampler = RandomSampler(self.dataset, shuffle)
        dataloader = DataLoader(
            self.dataset,
            batch_sampler=batch_sampler
        )
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, "dist", False)

        assert not (replaced_loader is dataloader)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
        assert not (replaced_loader.batch_sampler.sampler is dataloader.batch_sampler.sampler)
        assert replaced_loader.batch_sampler.batch_size == 2
        assert replaced_loader.batch_sampler.sampler.shuffle == shuffle
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        dist.barrier()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_set_dist_repro_dataloader_with_dist_dist_dataloader_normal(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'dist'、dataloader 为一般情况的表现
        """
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=shuffle)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, "dist", False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
        assert replaced_loader.batch_sampler.batch_size == dataloader.batch_sampler.batch_size
        assert replaced_loader.batch_sampler.sampler.shuffle == shuffle
        dist.barrier()

    """
    传入的参数 `dist` 为 'unrepeatdist' 的情况，这种情况出现在 evaluator 的初始化过程中，用户指定了 `use_dist_sampler` 参数
    为 True。此时函数会根据 dataloader 的  sampler 是否为 Unrepeated 和 Reproducible 来决定如何重新实例化 dataloader
    """

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_set_dist_repro_dataloader_with_dist_unrepeat_dataloader_reproducible_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'unrepeatdist'、dataloader.batch_sampler.sampler 为 ReproducibleSampler
        的表现
        """
        batch_sampler = BatchSampler(dataset=self.dataset, batch_size=2)
        batch_sampler.sampler = RandomSampler(self.dataset, shuffle)
        dataloader = DataLoader(
            self.dataset,
            batch_sampler=batch_sampler
        )
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, "unrepeatdist", False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, UnrepeatedRandomSampler)
        assert replaced_loader.batch_sampler.batch_size == 2
        assert replaced_loader.batch_sampler.sampler.shuffle == shuffle
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        dist.barrier()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_set_dist_repro_dataloader_with_dist_unrepeat_dataloader_unrepreated_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'unrepeatdist'、dataloader.batch_sampler.sampler 为 UnrepeatedSampler
        的表现
        """
        batch_sampler = BatchSampler(dataset=self.dataset, batch_size=2)
        batch_sampler.sampler = UnrepeatedRandomSampler(self.dataset, shuffle)
        dataloader = DataLoader(
            self.dataset,
            batch_sampler=batch_sampler
        )
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, "unrepeatdist", False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, UnrepeatedRandomSampler)
        assert not (replaced_loader.batch_sampler.sampler is dataloader.batch_sampler.sampler)
        assert replaced_loader.batch_sampler.batch_size == 2
        assert replaced_loader.drop_last == dataloader.drop_last
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        dist.barrier()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_set_dist_repro_dataloader_with_dist_unrepeat_dataloader_normal(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'unrepeatdist'、dataloader 为一般情况的表现
        """
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=shuffle)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, "unrepeatdist", False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, UnrepeatedSequentialSampler)
        assert replaced_loader.batch_sampler.batch_size == 4
        assert replaced_loader.drop_last == dataloader.drop_last
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        dist.barrier()

    def check_distributed_sampler(self, sampler):
        """
        测试替换得到的 sampler 或 batch_sampler 的分布式设置是否正确
        """
        assert sampler.num_replicas == dist.get_world_size()
        assert sampler.rank == dist.get_rank()
        if not isinstance(sampler, UnrepeatedSampler):
            assert sampler.pad == True


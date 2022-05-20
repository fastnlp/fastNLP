import pytest
from pathlib import Path

from fastNLP.core.drivers.paddle_driver.fleet import PaddleFleetDriver
from fastNLP.core.samplers import (
    RandomSampler,
    UnrepeatedSampler,
    BucketedBatchSampler,
    UnrepeatedRandomSampler,
    UnrepeatedSequentialSampler,
)
from tests.helpers.models.paddle_model import PaddleNormalModel_Classification_1
from tests.helpers.datasets.paddle_data import PaddleNormalDataset, PaddleRandomMaxDataset
from tests.helpers.utils import magic_argv_env_context
from fastNLP.envs.distributed import rank_zero_rm
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE
if _NEED_IMPORT_PADDLE:
    import paddle
    import paddle.distributed as dist
    from paddle.io import DataLoader, BatchSampler

def generate_driver(num_labels, feature_dimension, device=[0,1], fp16=False, output_from_new_proc="only_error"):
    paddle_model = PaddleNormalModel_Classification_1(num_labels, feature_dimension)
    paddle_opt = paddle.optimizer.Adam(parameters=paddle_model.parameters(), learning_rate=0.01)
    driver = PaddleFleetDriver(
        model=paddle_model,
        parallel_device=device,
        fp16=fp16,
        output_from_new_proc=output_from_new_proc
    )
    driver.set_optimizers(paddle_opt)
    driver.setup()

    return driver

############################################################################
#
# 测试 PaddleFleetDriver 的一些函数
#
############################################################################

@pytest.mark.paddledist
class TestFleetDriverFunction:
    """
    测试 PaddleFleetDriver 一些简单函数的测试类，基本都是测试能否运行、是否存在 import 错误等问题
    """

    @classmethod
    def setup_class(cls):
        cls.driver = generate_driver(10, 10)

    @magic_argv_env_context
    def test_multi_drivers(self):
        """
        测试使用了多个 PaddleFleetDriver 的情况。
        """
        driver2 = generate_driver(20, 10)

        with pytest.raises(RuntimeError):
            # 设备设置不同，应该报错
            driver3 = generate_driver(20, 3, device=[0,2])

        dist.barrier()

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
        res = self.driver.get_model_no_sync_context()
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

    @magic_argv_env_context
    def test_all_gather(self):
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

    @magic_argv_env_context
    @pytest.mark.parametrize("src_rank", ([0, 1]))
    def test_broadcast_object(self, src_rank):
        """
        测试 broadcast_object 函数
        详细的函数在 test_dist_utils.py 中完成
        """
        if self.driver.global_rank == src_rank:
            obj = {
                "rank": self.driver.global_rank
            }
        else:
            obj = None
        res = self.driver.broadcast_object(obj, src=src_rank)
        assert res["rank"] == src_rank

############################################################################
#
# 测试 set_dist_repro_dataloader 函数
#
############################################################################

@pytest.mark.paddledist
class TestSetDistReproDataloader:

    @classmethod
    def setup_class(cls):
        cls.device = [0, 1]
        cls.driver = generate_driver(10, 10, device=cls.device)

    def setup_method(self):
        self.dataset = PaddleNormalDataset(40)

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
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=not shuffle)
        batch_sampler = BucketedBatchSampler(self.dataset, self.dataset._data, batch_size=4, shuffle=shuffle)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, batch_sampler, False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BucketedBatchSampler)
        assert replaced_loader.batch_sampler is batch_sampler
        self.check_distributed_sampler(replaced_loader.batch_sampler)
        self.check_set_dist_repro_dataloader(dataloader, replaced_loader, shuffle)
        
        dist.barrier()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 RandomSampler 时的表现
        此时应该将 batch_sampler.sampler 替换为 dist 对应的 RandomSampler
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
        self.check_set_dist_repro_dataloader(dataloader, replaced_loader, shuffle)

        dist.barrier()
    
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
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=True)
        with pytest.raises(RuntimeError):
            # 应当抛出 RuntimeError
            replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, None, True)

        dist.barrier()

    @magic_argv_env_context
    # @pytest.mark.parametrize("shuffle", ([True, False]))
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_none_reproducible_false_dataloader_reproducible_batch_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 None、reproducible 为 False 、dataloader 有 BucketedBatchSampler 
        时的表现
        此时传入的 dataloader 的 batch_sampler 应该已经执行了 set_distributed，产生一个新的 dataloader，其 batch_sampler
        和原 dataloader 相同
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
        self.check_set_dist_repro_dataloader(dataloader, replaced_loader, shuffle)

        dist.barrier()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_none_reproducible_false_dataloader_reproducible_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 None、reproducible 为 False 、dataloader 有 RandomSampler 时的表现
        此时传入的 dataloader 的 batch_sampler.sampler 应该已经执行了 set_distributed，产生一个新的 dataloader，其
        batch_sampler.sampler 和原 dataloader 相同
        """
        batch_sampler = BatchSampler(dataset=self.dataset, batch_size=4)
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
        assert replaced_loader.batch_sampler.batch_size == 4
        assert replaced_loader.batch_sampler.drop_last == False
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        self.check_set_dist_repro_dataloader(dataloader, replaced_loader, shuffle)
    
        dist.barrier()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_none_reproducible_false_dataloader_normal(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 None、reproducible 为 False 、dataloader 为一般情况时的表现
        此时直接返回原来的 dataloader，不做任何处理。
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
    def test_with_dist_dist_dataloader_reproducible_batch_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'dist'、dataloader.batch_sampler 为 ReproducibleBatchSampler
        的表现
        此时应该返回一个新的 dataloader，其batch_sampler 和原 dataloader 相同，且应该正确地设置了分布式相关的属性
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
    def test_with_dist_dist_dataloader_reproducible_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'dist'、dataloader.batch_sampler.sampler 为 ReproducibleSampler
        的表现
        此时应该返回一个新的 dataloader，其 batch_sampler.sampler 和原 dataloader 相同，且应该正确地设置了分布式相关
        的属性
        """
        batch_sampler = BatchSampler(dataset=self.dataset, batch_size=4, shuffle=shuffle)
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
        assert replaced_loader.batch_sampler.batch_size == 4
        assert replaced_loader.batch_sampler.sampler.shuffle == shuffle
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        dist.barrier()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_dist_dataloader_normal(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'dist'、dataloader 为一般情况的表现
        此时应该返回一个新的 dataloader，并替换其 batch_sampler.sampler 为 RandomSampler，且应该正确设置了分布式相关
        的属性
        """
        dataloader = DataLoader(self.dataset, batch_size=4, shuffle=shuffle)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, "dist", False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
        assert replaced_loader.batch_sampler.batch_size == dataloader.batch_sampler.batch_size
        assert replaced_loader.batch_sampler.sampler.shuffle == shuffle
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        dist.barrier()

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
        batch_sampler = BatchSampler(dataset=self.dataset, batch_size=4)
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
        assert replaced_loader.batch_sampler.batch_size == 4
        assert replaced_loader.batch_sampler.sampler.shuffle == shuffle
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        dist.barrier()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_unrepeat_dataloader_unrepreated_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'unrepeatdist'、dataloader.batch_sampler.sampler 为 UnrepeatedSampler
        的表现
        此时应该返回一个新的 dataloader，且重新实例化了原来的 Sampler
        """
        batch_sampler = BatchSampler(dataset=self.dataset, batch_size=4)
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
        assert replaced_loader.batch_sampler.batch_size == 4
        assert replaced_loader.drop_last == dataloader.drop_last
        self.check_distributed_sampler(replaced_loader.batch_sampler.sampler)
        dist.barrier()

    @magic_argv_env_context
    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_unrepeat_dataloader_normal(self, shuffle):
        """
        测试 set_dist_repro_dataloader 中 dist 为 'unrepeatdist'、dataloader 为一般情况的表现
        此时应该返回一个新的 dataloader，且将 sampler 替换为 UnrepeatedSequentialSampler，并正确地设置了分布式相关
        的属性
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

    def check_set_dist_repro_dataloader(self, dataloader, replaced_loader, shuffle):
        """
        测试多卡下 set_dist_repro_dataloader 函数的执行结果是否正确
        """
        # 迭代两个 batch
        num_replicas = len(self.device)
        num_consumed_batches = 2
        already_seen_idx = set()
        for idx, batch in enumerate(replaced_loader):
            if idx >= num_consumed_batches:
                break
            already_seen_idx.update(batch)
        dist.barrier()
        if isinstance(replaced_loader.batch_sampler, BucketedBatchSampler):
            sampler_states = replaced_loader.batch_sampler.state_dict()
        else:
            sampler_states = replaced_loader.batch_sampler.sampler.state_dict()

        # 重新加载，应该可以输出剩下的内容，且对于 PaddleNormalDataset 来说，排序后应该是一个 range
        left_idxes = set()
        if isinstance(replaced_loader.batch_sampler, BucketedBatchSampler):
            batch_size = replaced_loader.batch_sampler.batch_size
            sampler_states["num_consumed_samples"] = num_consumed_batches * batch_size * num_replicas
            # 重新改造 dataloader
            new_loader = DataLoader(
                dataset=replaced_loader.dataset,
                batch_sampler=BucketedBatchSampler(
                    replaced_loader.dataset,
                    length=replaced_loader.dataset._data,
                    batch_size=batch_size,
                    shuffle=shuffle,
                )
            )
            new_loader.batch_sampler.set_distributed(
                num_replicas=self.driver.world_size,
                rank=self.driver.global_rank,
                pad=True
            )
            new_loader.batch_sampler.load_state_dict(sampler_states)
        else:
            batch_size = replaced_loader.batch_sampler.batch_size
            sampler_states["num_consumed_samples"] = num_consumed_batches * batch_size * num_replicas
            # 重新构造 dataloader
            batch_sampler = BatchSampler(replaced_loader.dataset, shuffle=shuffle, batch_size=batch_size)
            batch_sampler.sampler = RandomSampler(replaced_loader.dataset, shuffle=shuffle)
            batch_sampler.sampler.set_distributed(
                num_replicas=self.driver.world_size,
                rank=self.driver.global_rank
            )
            new_loader = DataLoader(replaced_loader.dataset, batch_sampler=batch_sampler)
            new_loader.batch_sampler.sampler.load_state_dict(sampler_states)
        for idx, batch in enumerate(new_loader):
            left_idxes.update(batch)

        assert len(left_idxes) + len(already_seen_idx) == len(self.dataset) / num_replicas
        assert len(left_idxes | already_seen_idx) == len(self.dataset) / num_replicas


############################################################################
#
# 测试 save 和 load 相关的功能
#
############################################################################

@pytest.mark.paddledist
class TestSaveLoad:
    """
    测试多卡情况下 save 和 load 相关函数的表现
    """

    @classmethod
    def setup_class(cls):
        # 不在这里 setup 的话会报错
        cls.driver = generate_driver(10, 10, device=[0,1])

    def setup_method(self):
        self.dataset = PaddleRandomMaxDataset(20, 10)

    @magic_argv_env_context
    @pytest.mark.parametrize("only_state_dict", ([True, False]))
    def test_save_and_load_model(self, only_state_dict):
        """
        测试 save_model 和 load_model 函数
        """
        try:
            path = "model"

            dataloader = DataLoader(self.dataset, batch_size=2)
            self.driver1, self.driver2 = generate_driver(10, 10), generate_driver(10, 10)

            if only_state_dict:
                self.driver1.save_model(path, only_state_dict)
            else:
                self.driver1.save_model(path, only_state_dict, input_spec=[paddle.ones((4, 10))])

            # 同步
            dist.barrier()
            self.driver2.load_model(path, only_state_dict)

            for idx, batch in enumerate(dataloader):
                batch = self.driver1.move_data_to_device(batch)
                res1 = self.driver1.model(
                    batch,
                    fastnlp_fn=self.driver1.model._layers.model.evaluate_step,
                    # Driver.model -> DataParallel._layers -> _FleetWrappingModel.model
                    fastnlp_signature_fn=None,
                    wo_auto_param_call=False,
                )
                res2 = self.driver2.model(
                    batch,
                    fastnlp_fn=self.driver2.model._layers.model.evaluate_step,
                    fastnlp_signature_fn=None,
                    wo_auto_param_call=False,
                )

                assert paddle.equal_all(res1["pred"], res2["pred"])
        finally:
            if only_state_dict:
                rank_zero_rm(path)
            else:
                rank_zero_rm(path + ".pdiparams")
                rank_zero_rm(path + ".pdiparams.info")
                rank_zero_rm(path + ".pdmodel")

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

            self.driver1, self.driver2 = generate_driver(10, 10, device=device, fp16=fp16), \
                                            generate_driver(10, 10, device=device, fp16=False)
            dataloader = DataLoader(
                dataset=self.dataset,
                batch_sampler=BucketedBatchSampler(
                    self.dataset,
                    length=[10 for i in range(len(self.dataset))],
                    batch_size=4,
                )
            )
            dataloader.batch_sampler.set_distributed(
                num_replicas=self.driver1.world_size,
                rank=self.driver1.global_rank,
                pad=True
            )
            num_consumed_batches = 2

            already_seen_x_set = set()
            already_seen_y_set = set()
            for idx, batch in enumerate(dataloader):
                if idx >= num_consumed_batches:
                    break
                already_seen_x_set.update(batch["x"])
                already_seen_y_set.update(batch["y"])

            # 同步
            dist.barrier()

            # 保存状态
            sampler_states = dataloader.batch_sampler.state_dict()
            save_states = {"num_consumed_batches": num_consumed_batches}
            if only_state_dict:
                self.driver1.save_checkpoint(Path(path), save_states, dataloader, only_state_dict, should_save_model=True)
            else:
                self.driver1.save_checkpoint(Path(path), save_states, dataloader, only_state_dict, should_save_model=True, input_spec=[paddle.ones((16, 10))])
            dist.barrier()
            # 加载
            # 更改 batch_size
            dataloader = DataLoader(
                dataset=self.dataset,
                batch_sampler=BucketedBatchSampler(
                    self.dataset,
                    length=[10 for i in range(len(self.dataset))],
                    batch_size=2,
                )
            )
            dataloader.batch_sampler.set_distributed(
                num_replicas=self.driver2.world_size,
                rank=self.driver2.global_rank,
                pad=True
            )
            dist.barrier()
            load_states = self.driver2.load_checkpoint(Path(path), dataloader, only_state_dict, should_load_model=True)
            replaced_loader = load_states.pop("dataloader")
            dist.barrier()
            # 1. 检查 optimizer 的状态
            # TODO optimizer 的 state_dict 总是为空

            # 2. 检查 batch_sampler 是否被正确地加载和替换
            assert not (replaced_loader is dataloader)
            assert replaced_loader.batch_sampler is dataloader.batch_sampler
            assert isinstance(replaced_loader.batch_sampler, BucketedBatchSampler)
            assert replaced_loader.batch_sampler.seed == sampler_states["seed"]
            assert replaced_loader.batch_sampler.num_consumed_samples == num_consumed_batches * 4 * num_replicas

            # 3. 检查 fp16 是否被加载
            if fp16:
                assert not isinstance(self.driver2.grad_scaler, paddle.amp.GradScaler)

            # 4. 检查 model 的参数是否正确
            # 5. 检查 batch_idx
            start_batch = load_states.pop('batch_idx_in_epoch')
            assert start_batch == 2 * num_consumed_batches
            left_x_batches = set()
            left_y_batches = set()
            for idx, batch in enumerate(replaced_loader):

                left_x_batches.update(batch["x"])
                left_y_batches.update(batch["y"])
                res1 = self.driver1.model(
                    batch,
                    fastnlp_fn=self.driver1.model._layers.model.evaluate_step,
                    # Driver.model -> DataParallel._layers -> _FleetWrappingModel.model
                    fastnlp_signature_fn=None,
                    wo_auto_param_call=False,
                )
                res2 = self.driver2.model(
                    batch,
                    fastnlp_fn=self.driver2.model._layers.model.evaluate_step,
                    fastnlp_signature_fn=None,
                    wo_auto_param_call=False,
                )
                assert paddle.equal_all(res1["pred"], res2["pred"])

            assert len(left_x_batches) + len(already_seen_x_set) == len(self.dataset) / num_replicas
            assert len(left_x_batches | already_seen_x_set) == len(self.dataset) / num_replicas
            assert len(left_y_batches) + len(already_seen_y_set) == len(self.dataset) / num_replicas
            assert len(left_y_batches | already_seen_y_set) == len(self.dataset) / num_replicas
        finally:
            rank_zero_rm(path)

    @magic_argv_env_context
    @pytest.mark.parametrize("only_state_dict", ([True, False]))
    @pytest.mark.parametrize("fp16", ([True, False]))
    @pytest.mark.parametrize("device", ([[0,1]]))
    def test_save_and_load_with_randomsampler(self, device, only_state_dict, fp16):
        """
        测试save和load函数，主要测试 dataloader 被替换了 batch_sampler 的情况
        """

        try:
            path = "model.ckp"

            num_replicas = len(device)

            self.driver1 = generate_driver(10, 10, device=device, fp16=fp16)
            self.driver2 = generate_driver(10, 10, device=device, fp16=False)
            batch_sampler = BatchSampler(dataset=self.dataset, batch_size=4)
            batch_sampler.sampler = RandomSampler(self.dataset, True)
            batch_sampler.sampler.set_distributed(
                num_replicas=self.driver1.world_size,
                rank=self.driver1.global_rank,
                pad=True
            )
            dataloader = DataLoader(
                self.dataset,
                batch_sampler=batch_sampler
            )
            num_consumed_batches = 2

            already_seen_x_set = set()
            already_seen_y_set = set()
            for idx, batch in enumerate(dataloader):
                if idx >= num_consumed_batches:
                    break
                already_seen_x_set.update(batch["x"])
                already_seen_y_set.update(batch["y"])

            # 同步
            dist.barrier()

            # 保存状态
            sampler_states = dataloader.batch_sampler.sampler.state_dict()
            save_states = {"num_consumed_batches": num_consumed_batches}
            if only_state_dict:
                self.driver1.save_checkpoint(Path(path), save_states, dataloader, only_state_dict, should_save_model=True)
            else:
                self.driver1.save_checkpoint(Path(path), save_states, dataloader, only_state_dict, should_save_model=True, input_spec=[paddle.ones((16, 10))])
            # 加载
            # 更改 batch_size
            batch_sampler = BatchSampler(dataset=self.dataset, batch_size=2)
            batch_sampler.sampler = RandomSampler(self.dataset, True)
            batch_sampler.sampler.set_distributed(
                num_replicas=self.driver2.world_size,
                rank=self.driver2.global_rank,
                pad=True
            )
            dataloader = DataLoader(
                self.dataset,
                batch_sampler=batch_sampler
            )
            load_states = self.driver2.load_checkpoint(Path(path), dataloader, only_state_dict, should_load_model=True)
            replaced_loader = load_states.pop("dataloader")

            # 1. 检查 optimizer 的状态
            # TODO optimizer 的 state_dict 总是为空

            # 2. 检查 sampler 是否被正确地加载和替换
            assert not (replaced_loader is dataloader)
            assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
            assert replaced_loader.batch_sampler.sampler.seed == sampler_states["seed"]
            assert replaced_loader.batch_sampler.sampler.epoch == sampler_states["epoch"]
            assert replaced_loader.batch_sampler.sampler.num_consumed_samples == 4 * num_consumed_batches * num_replicas
            assert len(replaced_loader.batch_sampler.sampler.dataset) == sampler_states["length"]
            assert replaced_loader.batch_sampler.sampler.shuffle == sampler_states["shuffle"]
            # 3. 检查 fp16 是否被加载
            if fp16:
                assert not isinstance(self.driver2.grad_scaler, paddle.amp.GradScaler)

            # 4. 检查 model 的参数是否正确
            # 5. 检查 batch_idx
            start_batch = load_states.pop('batch_idx_in_epoch')
            assert start_batch == 2 * num_consumed_batches
            left_x_batches = set()
            left_y_batches = set()
            for idx, batch in enumerate(replaced_loader):

                left_x_batches.update(batch["x"])
                left_y_batches.update(batch["y"])
                res1 = self.driver1.model(
                    batch,
                    fastnlp_fn=self.driver1.model._layers.model.evaluate_step,
                    # Driver.model -> DataParallel._layers -> _FleetWrappingModel.model
                    fastnlp_signature_fn=None,
                    wo_auto_param_call=False,
                )
                res2 = self.driver2.model(
                    batch,
                    fastnlp_fn=self.driver2.model._layers.model.evaluate_step,
                    fastnlp_signature_fn=None,
                    wo_auto_param_call=False,
                )
                assert paddle.equal_all(res1["pred"], res2["pred"])

            assert len(left_x_batches) + len(already_seen_x_set) == len(self.dataset) / num_replicas
            assert len(left_x_batches | already_seen_x_set) == len(self.dataset) / num_replicas
            assert len(left_y_batches) + len(already_seen_y_set) == len(self.dataset) / num_replicas
            assert len(left_y_batches | already_seen_y_set) == len(self.dataset) / num_replicas

        finally:
            rank_zero_rm(path)
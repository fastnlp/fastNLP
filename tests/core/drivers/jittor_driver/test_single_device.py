import pytest
from copy import deepcopy
from pathlib import Path

from fastNLP.core.drivers.jittor_driver import JittorSingleDriver
from fastNLP.core.samplers import ReproduceBatchSampler, RandomSampler
from fastNLP.core.dataloaders import JittorDataLoader
from tests.helpers.models.jittor_model import JittorNormalModel_Classification_1
from tests.helpers.datasets.jittor_data import JittorNormalDataset, JittorNormalXYDataset
from tests.helpers.datasets.torch_data import TorchNormalDataset
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from fastNLP.envs.distributed import rank_zero_rm
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR, _NEED_IMPORT_TORCH
if _NEED_IMPORT_JITTOR:
    import jittor as jt
    from jittor.dataset import (
        BatchSampler as JittorBatchSampler,
        RandomSampler as JittorRandomSampler,
        SequentialSampler as JittorSequentialSampler,
        SubsetRandomSampler as JittorSubsetRandomSampler
    )

if _NEED_IMPORT_TORCH:
    import torch

def get_dataloader(dataset, use_dataloader, sampler, batch_size, shuffle, drop_last=False):
    """
    :param dataset:
    :param use_dataloader: 是否使用 JittorDataLoader 包裹
    :param sampler: 使用 BatchSampler Samlper 还是不使用 Sampler
    """
    if use_dataloader:
        dataloader = JittorDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
        dataloader.dataset.set_attrs(sampler=sampler)
    else:
        dataloader = dataset
        dataloader.set_attrs(batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, sampler=sampler)

    return dataloader
############################################################################
#
# 测试基类 JittorDrvier 中的一些简单函数
#
############################################################################

class TestJittorDriverFunctions:
    """
    使用 JittorSingleDriver 测试基类的函数
    """

    @classmethod
    def setup_class(self):
        model = JittorNormalModel_Classification_1(10, 32)
        self.driver = JittorSingleDriver(model, device="cpu")

    @pytest.mark.jittor
    def test_check_optimizers_legality(self):
        """
        测试对合法的 optimizers 的检查
        """
        # 单个 optimizer
        optimizer = jt.optim.Adam(
            params=self.driver.model.parameters(),
            lr=0.01
        )
        self.driver.set_optimizers(optimizer)

        # optimizer 列表
        optimizers = [
            jt.optim.Adam(
                params=self.driver.model.parameters(),
                lr=0.01
            ) for i in range(10)
        ]
        self.driver.set_optimizers(optimizers)

    @pytest.mark.torchjittor
    def test_invalid_optimizers(self):
        """
        测试传入非法的 optimizers
        """
        # 单个 optimizer
        optimizer = torch.optim.Adam(TorchNormalModel_Classification_1(10, 32).parameters(), 0.01)
        with pytest.raises(TypeError):
            self.driver.set_optimizers(optimizer)

        optimizers = [
            torch.optim.Adam(TorchNormalModel_Classification_1(10, 32).parameters(), 0.01)
        ]

        with pytest.raises(TypeError):
            self.driver.set_optimizers(optimizers)

    @pytest.mark.jittor
    def test_check_dataloader_legality(self):
        """
        测试 check_dataloader_legality 函数的表现
        """
        # 使用 JittorDataLoader
        dataloader = JittorDataLoader(JittorNormalDataset())
        self.driver.check_dataloader_legality(dataloader)
        # 使用 jittor.dataset.Dataset
        self.driver.check_dataloader_legality(JittorNormalDataset())

    @pytest.mark.torchjittor
    def test_check_dataloader_legality_invalid(self):
        """
        测试 check_dataloader_legality 函数传入其他类型的表现
        """
        # 创建 torch 的 dataloader
        dataloader = torch.utils.data.DataLoader(
            TorchNormalDataset(),
            batch_size=32, shuffle=True
        )
        with pytest.raises(TypeError):
            self.driver.check_dataloader_legality(dataloader)

    @pytest.mark.jittor
    def test_tensor_to_numeric(self):
        """
        测试 tensor_to_numeric 函数
        """
        # 单个张量
        tensor = jt.Var(3)
        res = JittorSingleDriver.tensor_to_numeric(tensor)
        assert res == 3

        tensor = jt.rand(3, 4)
        res = JittorSingleDriver.tensor_to_numeric(tensor)
        assert res == tensor.tolist()

        # 张量list
        tensor_list = [jt.rand(6, 4, 2) for i in range(10)]
        res = JittorSingleDriver.tensor_to_numeric(tensor_list)
        assert isinstance(res, list)
        tensor_list = [t.tolist() for t in tensor_list]
        assert res == tensor_list

        # 张量tuple
        tensor_tuple = tuple([jt.rand(6, 4, 2) for i in range(10)])
        res = JittorSingleDriver.tensor_to_numeric(tensor_tuple)
        assert isinstance(res, tuple)
        tensor_tuple = tuple([t.tolist() for t in tensor_tuple])
        assert res == tensor_tuple

        # 张量dict
        tensor_dict = {
            "tensor": jt.rand(3, 4),
            "list": [jt.rand(6, 4, 2) for i in range(10)],
            "dict":{
                "list": [jt.rand(6, 4, 2) for i in range(10)],
                "tensor": jt.rand(3, 4)
            },
            "int": 2,
            "string": "test string"
        }

        res = JittorSingleDriver.tensor_to_numeric(tensor_dict)
        assert isinstance(res, dict)
        assert res["tensor"] == tensor_dict["tensor"].tolist()
        assert isinstance(res["list"], list)
        for r, d in zip(res["list"], tensor_dict["list"]):
            assert r == d.tolist()
        assert isinstance(res["int"], int)
        assert isinstance(res["string"], str)
        assert isinstance(res["dict"], dict)
        assert isinstance(res["dict"]["list"], list)
        for r, d in zip(res["dict"]["list"], tensor_dict["dict"]["list"]):
            assert r == d.tolist()
        assert res["dict"]["tensor"] == tensor_dict["dict"]["tensor"].tolist()

    @pytest.mark.jittor
    def test_tensor_to_numeric_reduce(self):
        tensor = jt.Var([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        res_max = JittorSingleDriver.tensor_to_numeric(tensor, reduce="max")
        res_min = JittorSingleDriver.tensor_to_numeric(tensor, reduce="min")
        res_sum = JittorSingleDriver.tensor_to_numeric(tensor, reduce="sum")
        res_mean = JittorSingleDriver.tensor_to_numeric(tensor, reduce="mean")

        assert res_max == 6
        assert res_min == 1
        assert res_sum == 21
        assert res_mean == 3.5


    @pytest.mark.jittor
    def test_set_model_mode(self):
        """
        测试 set_model_mode 函数
        """
        self.driver.set_model_mode("train")
        assert self.driver.model.is_training()
        self.driver.set_model_mode("eval")
        assert not self.driver.model.is_training()
        # 应该报错
        with pytest.raises(AssertionError):
            self.driver.set_model_mode("test")

    @pytest.mark.jittor
    def test_move_model_to_device_cpu(self):
        """
        测试 move_model_to_device 函数，仅测试能否运行
        """
        JittorSingleDriver.move_model_to_device(self.driver.model, "cpu")

    @pytest.mark.jittor
    def test_move_model_to_device_gpu(self):
        """
        测试 move_model_to_device 函数，仅测试能否运行
        """
        JittorSingleDriver.move_model_to_device(self.driver.model, "gpu")

    @pytest.mark.jittor
    def test_set_deterministic_dataloader(self):
        """
        测试 set_deterministic_dataloader，仅测试能否运行
        """
        # 先确保不影响运行
        # TODO：正确性
        dataloader = JittorDataLoader(JittorNormalDataset())
        self.driver.set_deterministic_dataloader(dataloader)
        self.driver.set_deterministic_dataloader(JittorNormalDataset())

    @pytest.mark.jittor
    def test_set_sampler_epoch(self):
        """
        测试 set_sampler_epoch
        """
        # 先确保不影响运行
        # TODO：正确性
        dataloader = JittorDataLoader(JittorNormalDataset())
        self.driver.set_sampler_epoch(dataloader, 0)
        self.driver.set_sampler_epoch(JittorNormalDataset(), 0)

    @pytest.mark.jittor
    @pytest.mark.parametrize("batch_size", [16])
    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("drop_last", [True, False])
    @pytest.mark.parametrize("use_dataloader", [True, False])
    def test_get_dataloader_args(self, batch_size, shuffle, drop_last, use_dataloader):
        """
        测试正常情况下 get_dataloader_args 的表现
        """
        dataloader = get_dataloader(
            JittorNormalDataset(),
            use_dataloader=use_dataloader,
            sampler=None,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last
        )
        res = JittorSingleDriver.get_dataloader_args(dataloader)

        assert isinstance(res.dataset, JittorNormalDataset)
        assert res.sampler is None
        assert res.shuffle == shuffle
        assert res.batch_size == batch_size
        assert res.drop_last == drop_last

    @pytest.mark.jittor
    @pytest.mark.parametrize("batch_size", [16])
    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("drop_last", [True, False])
    @pytest.mark.parametrize("use_dataloader", [True, False])
    def test_get_dataloader_args_with_randomsampler(self, batch_size, shuffle, drop_last, use_dataloader):
        """
        测试替换了 sampler 后 get_dataloader_args 的表现
        """
        dataset = JittorNormalDataset()
        dataloader = get_dataloader(
            dataset,
            use_dataloader=use_dataloader,
            batch_size=batch_size,
            sampler=RandomSampler(dataset, shuffle=shuffle),
            shuffle=shuffle,
            drop_last=drop_last
        )

        res = JittorSingleDriver.get_dataloader_args(dataloader)

        assert isinstance(res.dataset, JittorNormalDataset)
        assert isinstance(res.sampler, RandomSampler)
        assert res.shuffle == shuffle
        assert res.batch_size == batch_size
        assert res.drop_last == drop_last


############################################################################
#
# 测试 JittorSingleDrvier 中的一些简单函数
#
############################################################################

@pytest.mark.jittor
class TestSingleDeviceFunction:
    """
    测试其它函数的测试例
    """

    @classmethod
    def setup_class(cls):
        model = JittorNormalModel_Classification_1(10, 784)
        cls.driver = JittorSingleDriver(model, device="cpu")

    def test_unwrap_model(self):
        """
        测试能否运行
        """
        res = self.driver.unwrap_model()
        assert res is self.driver.model

    def test_is_distributed(self):
        assert self.driver.is_distributed() == False

    def test_move_data_to_device(self):
        self.driver.move_data_to_device(jt.rand(32, 64))


############################################################################
#
# 测试 set_dist_repro_dataloader 函数
#
############################################################################

@pytest.mark.jittor
class TestSetDistReproDataloader:
    """
    专门测试 set_dist_repro_dataloader 函数的类
    """
    def setup_method(self):
        self.dataset = JittorNormalDataset(20)
        model = JittorNormalModel_Classification_1(10, 32)
        self.driver = JittorSingleDriver(model, device="cpu")

    @pytest.mark.parametrize("use_dataloader", [True, False])
    def test_with_reproducible_false(self, use_dataloader):
        """
        测试 set_dist_repro_dataloader 参数 `reproducible` 为 False 时的表现
        当dist为字符串时，此时应该返回原来的 dataloader
        """
        dataloader = get_dataloader(self.dataset, use_dataloader, sampler=None, batch_size=2, shuffle=True)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist="dist", reproducible=False)

        assert replaced_loader is dataloader

    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("sampler", [None, "random", "sequential"])
    @pytest.mark.parametrize("use_dataloader", [True, False])
    def test_with_reproducible_true(self, shuffle, sampler, use_dataloader):
        """
        测试 set_dist_repro_dataloader 参数 `reproducible` 为 True 时的表现
        当dist为字符串时，此时应该返回新的 dataloader，会替换 sampler 为 RandomSampler
        """
        if sampler == "random":
            sampler = JittorRandomSampler(self.dataset)
            _shuffle = True
        elif sampler == "sequential":
            sampler = JittorSequentialSampler(self.dataset)
            _shuffle = False
        else:
            _shuffle = shuffle
        dataloader = get_dataloader(self.dataset, use_dataloader, sampler=sampler, batch_size=2, shuffle=shuffle)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist="dist", reproducible=True)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.sampler, RandomSampler)
        assert replaced_loader.sampler.shuffle == _shuffle
        assert replaced_loader.batch_size == dataloader.batch_size
        assert replaced_loader.drop_last == dataloader.drop_last

        self.check_set_dist_repro_dataloader(dataloader, replaced_loader, shuffle, use_dataloader)

    @pytest.mark.parametrize("shuffle", ([True, False]))
    @pytest.mark.parametrize("use_dataloader", [True, False])
    def test_with_dist_batch_sampler(self, shuffle, use_dataloader):
        """
        测试 set_dist_repro_dataloader 参数 dist 不是字符串时的表现，且 dist 是 ReproducibleBatchSampler
        应该返回新的 dataloader，并将 batch_sampler 替换为 dist 对应的 Sampler
        jittor 暂时不支持这种情况，会报错
        """
        dataloader = get_dataloader(self.dataset, use_dataloader, sampler=None, batch_size=2, shuffle=not shuffle)
        dist = ReproduceBatchSampler(JittorBatchSampler(JittorRandomSampler(self.dataset), 4, False), 4, False)

        with pytest.raises(RuntimeError):
            replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist=dist, reproducible=False)

    @pytest.mark.parametrize("shuffle", ([True, False]))
    @pytest.mark.parametrize("use_dataloader", [True, False])
    def test_with_dist_sampler(self, shuffle, use_dataloader):
        """
        测试 set_dist_repro_dataloader 参数 dist 不是字符串时的表现
        应该返回新的 dataloader，并将 sampler 替换为 dist 对应的 Sampler
        """
        dataloader = get_dataloader(self.dataset, use_dataloader, sampler=None, batch_size=2, shuffle=not shuffle)
        dist = RandomSampler(self.dataset, shuffle=shuffle)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist=dist, reproducible=False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.sampler, RandomSampler)
        assert replaced_loader.sampler is dist
        assert replaced_loader.batch_size == dataloader.batch_size

        self.check_set_dist_repro_dataloader(dataloader, replaced_loader, shuffle, use_dataloader)

    @pytest.mark.parametrize("shuffle", ([True, False]))
    @pytest.mark.parametrize("use_dataloader", [True, False])
    def test_with_dataloader_reproducible_batch_sampler(self, shuffle, use_dataloader):
        """
        测试 set_dist_repro_dataloader 参数 dataloader 已经支持断点重训时的表现
        应该返回新的 dataloader，且其余各项设置和原来相同
        """
        dataloader = get_dataloader(
            self.dataset, 
            use_dataloader=use_dataloader,
            sampler=ReproduceBatchSampler(
                        JittorBatchSampler(JittorRandomSampler(self.dataset), 4, False),
                        batch_size=4,
                        drop_last=False,
                    ),
            batch_size=4,
            shuffle=shuffle,
        )
        with pytest.raises(RuntimeError):
            replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist="dist", reproducible=False)

    @pytest.mark.parametrize("shuffle", ([True, False]))
    @pytest.mark.parametrize("use_dataloader", [True, False])
    def test_with_dataloader_reproducible_sampler(self, shuffle, use_dataloader):
        """
        测试 set_dist_repro_dataloader 参数 dataloader 已经支持断点重训时的表现
        应该返回新的 dataloader，且其余各项设置和原来相同
        """
        dataloader = get_dataloader(
            self.dataset, 
            use_dataloader=use_dataloader,
            sampler=RandomSampler(self.dataset, shuffle),
            batch_size=2,
            shuffle=shuffle,
        )
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist="dist", reproducible=False)

        assert not (replaced_loader is dataloader)
        assert not (replaced_loader.sampler is dataloader.sampler)
        assert isinstance(replaced_loader.sampler, RandomSampler)
        assert replaced_loader.batch_size == 2
        assert replaced_loader.shuffle == shuffle

        self.check_set_dist_repro_dataloader(dataloader, replaced_loader, shuffle, use_dataloader)

    def check_set_dist_repro_dataloader(self, dataloader, replaced_loader, shuffle, use_dataloader):
        """
        测试单卡下 set_dist_repro_dataloader 函数的执行结果是否正确
        """
        # 迭代两个 batch
        num_consumed_batches = 2
        already_seen_idx = set()
        replaced_loader.sampler.set_epoch(6)
        for idx, batch in enumerate(replaced_loader):
            if idx >= num_consumed_batches:
                break
            already_seen_idx.update(batch.tolist())
        sampler_states = replaced_loader.sampler.state_dict()

        # 重新加载，应该可以输出剩下的内容，且对于 JittorNormalDataset 来说，排序后应该是一个 range
        left_idxes = set()
        batch_size = replaced_loader.batch_size
        sampler_states["num_consumed_samples"] = num_consumed_batches * batch_size
        # 重新构造 dataloader
        if use_dataloader:
            dataset = deepcopy(replaced_loader.dataset.dataset)
        else:
            dataset = deepcopy(replaced_loader)
        new_loader = get_dataloader(
            dataset=dataset,
            use_dataloader=use_dataloader,
            sampler = RandomSampler(dataset, shuffle=shuffle),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False
        )
        new_loader.sampler.load_state_dict(sampler_states)
        new_loader.sampler.set_epoch(6)
        for idx, batch in enumerate(new_loader):
            left_idxes.update(batch.tolist())

        print(already_seen_idx)
        print(left_idxes)

        assert len(left_idxes) + len(already_seen_idx) == self.dataset.total_len
        assert len(left_idxes | already_seen_idx) == self.dataset.total_len

############################################################################
#
# 测试 save 和 load 相关的功能
#
############################################################################

def generate_random_driver(labels, features, fp16=False, device="cpu", lr=0.01):
    """
    生成driver
    """
    model = JittorNormalModel_Classification_1(labels, features)
    opt = jt.optim.Adam(params=model.parameters(), lr=lr)
    driver = JittorSingleDriver(model, device=device, fp16=fp16)
    driver.set_optimizers(opt)
    driver.setup()

    return driver

@pytest.mark.jittor
@pytest.mark.parametrize("only_state_dict", ([True, False]))
@pytest.mark.parametrize("use_dataloader", [True, False])
def test_save_and_load_model(only_state_dict, use_dataloader):
    """
    测试 save_model 和 load_model 函数
    """
    try:
        path = "model"
        dataset = JittorNormalXYDataset(20)
        dataloader = get_dataloader(dataset, sampler=None, use_dataloader=use_dataloader, batch_size=4, shuffle=True)
        driver1, driver2 = generate_random_driver(20, 1, device="gpu"), generate_random_driver(20, 1, device="gpu")

        driver1.save_model(path, only_state_dict)
        driver2.load_model(path, only_state_dict)

        for batch in dataloader:
            batch = driver1.move_data_to_device(batch)
            res1 = driver1.model.evaluate_step(**batch)
            res2 = driver2.model.evaluate_step(**batch)

            assert jt.all_(jt.equal(res1["pred"], res2["pred"]))
    finally:
        rank_zero_rm(path)

@pytest.mark.jittor
@pytest.mark.parametrize("only_state_dict", ([True, False]))
@pytest.mark.parametrize("use_dataloader", [True, False])
def test_save_and_load_with_randomsampler(only_state_dict, use_dataloader):
    """
    测试save和load函数，主要测试 dataloader 被替换了 sampler 的情况
    """

    try:
        path = "model.ckp"

        driver1, driver2 = generate_random_driver(20, 1, device="gpu", lr=0.01), \
                            generate_random_driver(20, 1, device="gpu", lr=0.001)
        dataset = JittorNormalXYDataset(20)
        dataloader = get_dataloader(
            dataset, use_dataloader,
            sampler = RandomSampler(dataset, True),
            batch_size=4,
            shuffle=True
        )
        num_consumed_batches = 2

        already_seen_x_set = set()
        already_seen_y_set = set()
        driver1.set_sampler_epoch(dataloader, 7)
        for idx, batch in enumerate(dataloader):
            if idx >= num_consumed_batches:
                break
            already_seen_x_set.update(batch["x"].reshape(-1, ).tolist())
            already_seen_y_set.update(batch["y"].reshape(-1, ).tolist())

        sampler_states = dataloader.sampler.state_dict()
        save_states = {"num_consumed_batches": num_consumed_batches}
        driver1.save_checkpoint(Path(path), save_states, dataloader, only_state_dict, should_save_model=True)
        
        # 加载
        # 更改 batch_size
        dataloader = get_dataloader(
            dataset, use_dataloader,
            sampler=RandomSampler(dataset, True),
            batch_size=2,
            shuffle=True
        )
        load_states = driver2.load_checkpoint(Path(path), dataloader, only_state_dict, should_load_model=True)
        replaced_loader = load_states.pop("dataloader")

        # 1. 检查 optimizer 的状态
        assert driver2.optimizers[0].lr == driver1.optimizers[0].lr

        # 2. 检查 sampler 是否被正确地加载和替换
        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.sampler, RandomSampler)
        assert replaced_loader.sampler.seed == sampler_states["seed"]
        assert replaced_loader.sampler.epoch == sampler_states["epoch"]
        assert replaced_loader.sampler.num_consumed_samples == 4 * num_consumed_batches
        assert replaced_loader.sampler.dataset.total_len == sampler_states["length"]
        assert replaced_loader.sampler.shuffle == sampler_states["shuffle"]

        # 4. 检查 model 的参数是否正确
        # 5. 检查 batch_idx
        start_batch = load_states.pop('batch_idx_in_epoch')
        assert start_batch == 2 * num_consumed_batches
        left_x_batches = set()
        left_y_batches = set()
        driver2.set_sampler_epoch(replaced_loader, 7)
        for idx, batch in enumerate(replaced_loader):

            left_x_batches.update(batch["x"].reshape(-1, ).tolist())
            left_y_batches.update(batch["y"].reshape(-1, ).tolist())
            res1 = driver1.model.evaluate_step(**batch)
            res2 = driver2.model.evaluate_step(**batch)
            assert jt.all_(jt.equal(res1["pred"], res2["pred"]))

        assert len(left_x_batches) + len(already_seen_x_set) == dataset.total_len
        assert len(left_x_batches | already_seen_x_set) == dataset.total_len
        assert len(left_y_batches) + len(already_seen_y_set) == dataset.total_len
        assert len(left_y_batches | already_seen_y_set) == dataset.total_len
    finally:
        rank_zero_rm(path)

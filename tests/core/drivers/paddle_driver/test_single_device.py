from dataclasses import replace
import os
from re import S
os.environ["FASTNLP_BACKEND"] = "paddle"
import pytest
from pathlib import Path

from fastNLP.core.drivers.paddle_driver.single_device import PaddleSingleDriver
from fastNLP.core.samplers import RandomBatchSampler, RandomSampler
from tests.helpers.models.paddle_model import PaddleNormalModel_Classification_1
from tests.helpers.datasets.paddle_data import PaddleNormalDataset, PaddleRandomMaxDataset
from tests.helpers.datasets.torch_data import TorchNormalDataset
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from fastNLP.core import synchronize_safe_rm

import paddle
from paddle.io import DataLoader, BatchSampler
import torch

############################################################################
#
# 测试基类 PaddleDrvier 中的一些简单函数
#
############################################################################

class TestPaddleDriverFunctions:
    """
    使用 PaddleSingleDriver 测试基类的函数
    """

    @classmethod
    def setup_class(self):
        model = PaddleNormalModel_Classification_1(10, 32)
        self.driver = PaddleSingleDriver(model, device="cpu")

    def test_check_single_optimizer_legality(self):
        """
        测试传入单个optimizer时的表现
        """
        optimizer = paddle.optimizer.Adam(
            parameters=self.driver.model.parameters(),
            learning_rate=0.01
        )

        self.driver.set_optimizers(optimizer)

        optimizer = torch.optim.Adam(TorchNormalModel_Classification_1(10, 32).parameters(), 0.01)
        # 传入torch的optimizer时，应该报错ValueError
        with pytest.raises(ValueError):
            self.driver.set_optimizers(optimizer)

    def test_check_optimizers_legality(self):
        """
        测试传入optimizer list的表现
        """
        optimizers = [
            paddle.optimizer.Adam(
                parameters=self.driver.model.parameters(),
                learning_rate=0.01
            ) for i in range(10)
        ]

        self.driver.set_optimizers(optimizers)

        optimizers += [
            torch.optim.Adam(TorchNormalModel_Classification_1(10, 32).parameters(), 0.01)
        ]

        with pytest.raises(ValueError):
            self.driver.set_optimizers(optimizers)

    def test_check_dataloader_legality_in_train(self):
        """
        测试is_train参数为True时，_check_dataloader_legality函数的表现
        """
        dataloader = paddle.io.DataLoader(PaddleNormalDataset())
        PaddleSingleDriver.check_dataloader_legality(dataloader, "dataloader", True)

        # batch_size 和 batch_sampler 均为 None 的情形
        dataloader = paddle.io.DataLoader(PaddleNormalDataset(), batch_size=None)
        with pytest.raises(ValueError):
            PaddleSingleDriver.check_dataloader_legality(dataloader, "dataloader", True)

        # 创建torch的dataloader
        dataloader = torch.utils.data.DataLoader(
            TorchNormalDataset(),
            batch_size=32, shuffle=True
        )
        with pytest.raises(ValueError):
            PaddleSingleDriver.check_dataloader_legality(dataloader, "dataloader", True)

    def test_check_dataloader_legality_in_test(self):
        """
        测试is_train参数为False时，_check_dataloader_legality函数的表现
        """
        # 此时传入的应该是dict
        dataloader = {
            "train": paddle.io.DataLoader(PaddleNormalDataset()),
            "test":paddle.io.DataLoader(PaddleNormalDataset())
        }
        PaddleSingleDriver.check_dataloader_legality(dataloader, "dataloader", False)

        # batch_size 和 batch_sampler 均为 None 的情形
        dataloader = {
            "train": paddle.io.DataLoader(PaddleNormalDataset()),
            "test":paddle.io.DataLoader(PaddleNormalDataset(), batch_size=None)
        }
        with pytest.raises(ValueError):
            PaddleSingleDriver.check_dataloader_legality(dataloader, "dataloader", False)

        # 传入的不是dict，应该报错
        dataloader = paddle.io.DataLoader(PaddleNormalDataset())
        with pytest.raises(ValueError):
            PaddleSingleDriver.check_dataloader_legality(dataloader, "dataloader", False)

        # 创建torch的dataloader
        train_loader = torch.utils.data.DataLoader(
            TorchNormalDataset(),
            batch_size=32, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            TorchNormalDataset(),
            batch_size=32, shuffle=True
        )
        dataloader = {"train": train_loader, "test": test_loader}
        with pytest.raises(ValueError):
            PaddleSingleDriver.check_dataloader_legality(dataloader, "dataloader", False)

    def test_tensor_to_numeric(self):
        """
        测试tensor_to_numeric函数
        """
        # 单个张量
        tensor = paddle.to_tensor(3)
        res = PaddleSingleDriver.tensor_to_numeric(tensor)
        assert res == 3

        tensor = paddle.rand((3, 4))
        res = PaddleSingleDriver.tensor_to_numeric(tensor)
        assert res == tensor.tolist()

        # 张量list
        tensor_list = [paddle.rand((6, 4, 2)) for i in range(10)]
        res = PaddleSingleDriver.tensor_to_numeric(tensor_list)
        assert isinstance(res, list)
        tensor_list = [t.tolist() for t in tensor_list]
        assert res == tensor_list

        # 张量tuple
        tensor_tuple = tuple([paddle.rand((6, 4, 2)) for i in range(10)])
        res = PaddleSingleDriver.tensor_to_numeric(tensor_tuple)
        assert isinstance(res, tuple)
        tensor_tuple = tuple([t.tolist() for t in tensor_tuple])
        assert res == tensor_tuple

        # 张量dict
        tensor_dict = {
            "tensor": paddle.rand((3, 4)),
            "list": [paddle.rand((6, 4, 2)) for i in range(10)],
            "dict":{
                "list": [paddle.rand((6, 4, 2)) for i in range(10)],
                "tensor": paddle.rand((3, 4))
            },
            "int": 2,
            "string": "test string"
        }

        res = PaddleSingleDriver.tensor_to_numeric(tensor_dict)
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

    def test_set_model_mode(self):
        """
        测试set_model_mode函数
        """
        self.driver.set_model_mode("train")
        assert self.driver.model.training
        self.driver.set_model_mode("eval")
        assert not self.driver.model.training
        # 应该报错
        with pytest.raises(AssertionError):
            self.driver.set_model_mode("test")

    def test_move_model_to_device_cpu(self):
        """
        测试move_model_to_device函数
        """
        PaddleSingleDriver.move_model_to_device(self.driver.model, "cpu")
        assert self.driver.model.linear1.weight.place.is_cpu_place()

    def test_move_model_to_device_gpu(self):
        """
        测试move_model_to_device函数
        """
        PaddleSingleDriver.move_model_to_device(self.driver.model, "gpu")
        assert self.driver.model.linear1.weight.place.is_gpu_place()
        assert self.driver.model.linear1.weight.place.gpu_device_id() == 0

    def test_worker_init_function(self):
        """
        测试worker_init_function
        """
        # 先确保不影响运行
        # TODO：正确性
        PaddleSingleDriver.worker_init_function(0)

    def test_set_deterministic_dataloader(self):
        """
        测试set_deterministic_dataloader
        """
        # 先确保不影响运行
        # TODO：正确性
        dataloader = DataLoader(PaddleNormalDataset())
        self.driver.set_deterministic_dataloader(dataloader)

    def test_set_sampler_epoch(self):
        """
        测试set_sampler_epoch
        """
        # 先确保不影响运行
        # TODO：正确性
        dataloader = DataLoader(PaddleNormalDataset())
        self.driver.set_sampler_epoch(dataloader, 0)

    @pytest.mark.parametrize("batch_size", [16])
    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("drop_last", [True, False])
    def test_get_dataloader_args(self, batch_size, shuffle, drop_last):
        """
        测试正常情况下 get_dataloader_args 的表现
        """
        dataloader = DataLoader(
            PaddleNormalDataset(),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        res = PaddleSingleDriver.get_dataloader_args(dataloader)

        assert isinstance(res.dataset, PaddleNormalDataset)
        assert isinstance(res.batch_sampler, BatchSampler)
        if shuffle:
            assert isinstance(res.sampler, paddle.io.RandomSampler)
        else:
            assert isinstance(res.sampler, paddle.io.SequenceSampler)
        assert res.shuffle == shuffle
        assert res.batch_size == batch_size
        assert res.drop_last == drop_last

    @pytest.mark.parametrize("batch_size", [16])
    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("drop_last", [True, False])
    def test_get_dataloader_args_with_randombatchsampler(self, batch_size, shuffle, drop_last):
        """
        测试替换了 batch_sampler 后 get_dataloader_args 的表现
        """
        dataset = PaddleNormalDataset()
        dataloader = DataLoader(
            dataset,
            batch_sampler=RandomBatchSampler(
                BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle),
                batch_size, 
                drop_last,
            )
        )
        res = PaddleSingleDriver.get_dataloader_args(dataloader)

        assert isinstance(res.dataset, PaddleNormalDataset)
        assert isinstance(res.batch_sampler, RandomBatchSampler)
        if shuffle:
            assert isinstance(res.sampler, paddle.io.RandomSampler)
        else:
            assert isinstance(res.sampler, paddle.io.SequenceSampler)
        assert res.shuffle == shuffle
        assert res.batch_size == batch_size
        assert res.drop_last == drop_last

    @pytest.mark.parametrize("batch_size", [16])
    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("drop_last", [True, False])
    def test_get_dataloader_args_with_randomsampler(self, batch_size, shuffle, drop_last):
        """
        测试替换了 sampler 后 get_dataloader_args 的表现
        """
        dataset = PaddleNormalDataset()
        batch_sampler = BatchSampler(dataset, batch_size=batch_size, drop_last=drop_last)
        batch_sampler.sampler = RandomSampler(dataset, shuffle)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler,
        )
        res = PaddleSingleDriver.get_dataloader_args(dataloader)

        assert isinstance(res.dataset, PaddleNormalDataset)
        assert isinstance(res.batch_sampler, BatchSampler)
        assert isinstance(res.sampler, RandomSampler)
        assert res.shuffle == shuffle
        assert res.batch_size == batch_size
        assert res.drop_last == drop_last


############################################################################
#
# 测试 PaddleSingleDrvier 中的一些简单函数
#
############################################################################

class TestSingleDeviceFunction:
    """
    测试其它函数的测试例
    """

    @classmethod
    def setup_class(cls):
        model = PaddleNormalModel_Classification_1(10, 784)
        cls.driver = PaddleSingleDriver(model, device="cpu")

    def test_unwrap_model(self):
        """
        测试能否运行
        """
        res = self.driver.unwrap_model()
        assert res is self.driver.model

    def test_is_distributed(self):
        assert self.driver.is_distributed() == False

    def test_move_data_to_device(self):
        """
        这个函数仅调用了paddle_move_data_to_device，测试例在tests/core/utils/test_paddle_utils.py中
        就不重复测试了
        """
        self.driver.move_data_to_device(paddle.rand((32, 64)))


############################################################################
#
# 测试 set_dist_repro_dataloader 函数
#
############################################################################

class TestSetDistReproDataloder:
    """
    专门测试 set_dist_repro_dataloader 函数的类
    """
    def setup_method(self):
        self.dataset = PaddleNormalDataset(20)
        model = PaddleNormalModel_Classification_1(10, 32)
        self.driver = PaddleSingleDriver(model, device="cpu")
    
    def test_set_dist_repro_dataloader_with_reproducible_false(self):
        """
        测试 set_dist_repro_dataloader 参数 `reproducible` 为 False 时的表现
        当dist为字符串时，此时应该返回原来的 dataloader
        """
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist="dist", reproducible=False)

        assert replaced_loader is dataloader

    @pytest.mark.parametrize("shuffle", [True, False])
    def test_set_dist_repro_dataloader_with_reproducible_true(self, shuffle):
        """
        测试 set_dist_repro_dataloader 参数 `reproducible` 为 True 时的表现
        当dist为字符串时，此时应该返回新的 dataloader，且如果原 sampler 为 paddle.io.RandomSampler（shuffle=True），
        只会替换 Sampler 为 RandomSampler；否则会替换 batch_sampler 为 RandomBatchSampler
        """
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=shuffle)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist="dist", reproducible=True)

        assert not (replaced_loader is dataloader)
        if shuffle:
            # 此时会替换 sampler
            assert isinstance(replaced_loader.batch_sampler, paddle.io.BatchSampler)
            assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
            assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
        else:
            # 此时会替换 batch_sampler
            assert isinstance(replaced_loader.batch_sampler, RandomBatchSampler)
            assert isinstance(replaced_loader.batch_sampler.batch_sampler, BatchSampler)
        assert replaced_loader.batch_sampler.batch_size == dataloader.batch_sampler.batch_size
        assert replaced_loader.drop_last == dataloader.drop_last

        self.check_set_dist_repro_dataloader(dataloader, replaced_loader, shuffle)

    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_set_dist_repro_dataloader_with_dist_batch_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 参数 dist 不是字符串时的表现，且 dist 是 ReproducibleBatchSampler
        应该返回新的 dataloader，并将 batch_sampler 替换为 dist 对应的 Sampler
        """
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=not shuffle)
        dist = RandomBatchSampler(BatchSampler(self.dataset, batch_size=4, shuffle=shuffle), 4, False)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist=dist, reproducible=False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, RandomBatchSampler)
        assert replaced_loader.batch_sampler is dist

        self.check_set_dist_repro_dataloader(dataloader, replaced_loader, shuffle)

    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_set_dist_repro_dataloader_with_dist_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 参数 dist 不是字符串时的表现
        应该返回新的 dataloader，并将 batch_sampler.sampler 替换为 dist 对应的 Sampler
        """
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=not shuffle)
        dist = RandomSampler(self.dataset, shuffle=shuffle)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist=dist, reproducible=False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, BatchSampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert replaced_loader.batch_sampler.sampler is dist
        assert replaced_loader.batch_sampler.batch_size == dataloader.batch_sampler.batch_size

        self.check_set_dist_repro_dataloader(dataloader, replaced_loader, shuffle)

    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_set_dist_repro_dataloader_with_dataloader_reproducible_batch_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 参数 dataloader 已经支持断点重训时的表现
        应该返回新的 dataloader，且其余各项设置和原来相同
        """
        dataloader = DataLoader(
            dataset=self.dataset,
            batch_sampler=RandomBatchSampler(
                BatchSampler(self.dataset, batch_size=4, shuffle=shuffle),
                batch_size=4,
                drop_last=False,
            )
        )
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist="dist", reproducible=False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, RandomBatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert replaced_loader.batch_sampler.batch_size == dataloader.batch_sampler.batch_size
        assert replaced_loader.drop_last == dataloader.drop_last

        self.check_set_dist_repro_dataloader(dataloader, replaced_loader, shuffle)

    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_set_dist_repro_dataloader_with_dataloader_reproducible_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 参数 dataloader 已经支持断点重训时的表现
        应该返回新的 dataloader，且其余各项设置和原来相同
        """
        batch_sampler = BatchSampler(dataset=self.dataset, batch_size=2, shuffle=shuffle)
        batch_sampler.sampler = RandomSampler(self.dataset, shuffle)
        dataloader = DataLoader(
            self.dataset,
            batch_sampler=batch_sampler
        )
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist="dist", reproducible=False)

        assert not (replaced_loader is dataloader)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
        assert not (replaced_loader.batch_sampler.sampler is dataloader.batch_sampler.sampler)
        assert replaced_loader.batch_sampler.batch_size == 2
        assert replaced_loader.batch_sampler.sampler.shuffle == shuffle

        self.check_set_dist_repro_dataloader(dataloader, replaced_loader, shuffle)

    def check_set_dist_repro_dataloader(self, dataloader, replaced_loader, shuffle):
        """
        测试单卡下 set_dist_repro_dataloader 函数的执行结果是否正确
        """
        # 迭代两个 batch
        num_consumed_batches = 2
        already_seen_idx = set()
        for idx, batch in enumerate(replaced_loader):
            if idx >= num_consumed_batches:
                break
            already_seen_idx.update(batch)
        if isinstance(replaced_loader.batch_sampler, RandomBatchSampler):
            sampler_states = replaced_loader.batch_sampler.state_dict()
        else:
            sampler_states = replaced_loader.batch_sampler.sampler.state_dict()

        # 加载 num_consumed_samples_array，设置正确取出的 batch 数目
        num_consumed_samples_array = sampler_states.pop('num_consumed_samples_array', None)

        # 重新加载，应该可以输出剩下的内容，且对于 PaddleNormalDataset 来说，排序后应该是一个 range
        left_idxes = set()
        if isinstance(replaced_loader.batch_sampler, RandomBatchSampler):
            batch_size = replaced_loader.batch_sampler.batch_size
            if num_consumed_samples_array is not None:
                sampler_states["num_consumed_samples"] = num_consumed_samples_array[num_consumed_batches]
            else:
                sampler_states["num_consumed_samples"] = num_consumed_batches * batch_size
            # 重新改造 dataloader
            new_loader = DataLoader(
                dataset=replaced_loader.dataset,
                batch_sampler=RandomBatchSampler(
                    BatchSampler(replaced_loader.dataset, shuffle=shuffle, batch_size=batch_size),
                    batch_size=batch_size,
                    drop_last=False,
                )
            )
            new_loader.batch_sampler.load_state_dict(sampler_states)
        else:
            batch_size = replaced_loader.batch_sampler.batch_size
            num_consumed_batches = num_consumed_batches * batch_size
            if num_consumed_samples_array is not None:
                sampler_states["num_consumed_samples"] = num_consumed_samples_array[num_consumed_batches]
            else:
                sampler_states["num_consumed_samples"] = num_consumed_batches * batch_size
            # 重新构造 dataloader
            batch_sampler = BatchSampler(replaced_loader.dataset, shuffle=shuffle, batch_size=batch_size)
            batch_sampler.sampler = RandomSampler(replaced_loader.dataset, shuffle=shuffle)
            new_loader = DataLoader(replaced_loader.dataset, batch_sampler=batch_sampler)
            new_loader.batch_sampler.sampler.load_state_dict(sampler_states)
        for idx, batch in enumerate(new_loader):
            left_idxes.update(batch)

        assert len(left_idxes) + len(already_seen_idx) == len(self.dataset)
        assert len(left_idxes | already_seen_idx) == len(self.dataset)

############################################################################
#
# 测试 save 和 load 相关的功能
#
############################################################################

def generate_random_driver(features, labels):
    """
    生成driver
    """
    model = PaddleNormalModel_Classification_1(labels, features)
    opt = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=0.01)
    driver = PaddleSingleDriver(model, device="cpu")
    driver.set_optimizers(opt)
    driver.setup()

    return driver

@pytest.fixture
def prepare_test_save_load():
    dataset = PaddleRandomMaxDataset(320, 10)
    dataloader = DataLoader(dataset, batch_size=32)
    driver1, driver2 = generate_random_driver(10, 10), generate_random_driver(10, 10)
    return driver1, driver2, dataloader

@pytest.mark.parametrize("only_state_dict", ([True, False]))
def test_save_and_load_model(prepare_test_save_load, only_state_dict):
    """
    测试 save_model 和 load_model 函数
    """
    try:
        path = "model"
        driver1, driver2, dataloader = prepare_test_save_load

        if only_state_dict:
            driver1.save_model(path, only_state_dict)
        else:
            driver1.save_model(path, only_state_dict, input_spec=[paddle.ones((32, 10))])
        driver2.load_model(path, only_state_dict)

        for batch in dataloader:
            batch = driver1.move_data_to_device(batch)
            res1 = driver1.model.evaluate_step(**batch)
            res2 = driver2.model.evaluate_step(**batch)

            assert paddle.equal_all(res1["pred"], res2["pred"])
    finally:
        if only_state_dict:
            synchronize_safe_rm(path)
        else:
            synchronize_safe_rm(path + ".pdiparams")
            synchronize_safe_rm(path + ".pdiparams.info")
            synchronize_safe_rm(path + ".pdmodel")

@pytest.mark.parametrize("only_state_dict", ([True, False]))
def test_save_and_load_with_randombatchsampler(only_state_dict):
    """
    测试save和load函数，主要测试 dataloader 被替换了 sampler 之后的情况
    """

    try:
        path = "model.ckp"

        driver1, driver2 = generate_random_driver(10, 10), generate_random_driver(10, 10)
        dataset = PaddleRandomMaxDataset(40, 10)
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=RandomBatchSampler(BatchSampler(dataset, batch_size=4), 4, False)
        )
        num_consumed_batches = 2

        already_seen_x_set = set()
        already_seen_y_set = set()
        for idx, batch in enumerate(dataloader):
            if idx >= num_consumed_batches:
                break
            already_seen_x_set.update(batch["x"])
            already_seen_y_set.update(batch["y"])

        sampler_states = dataloader.batch_sampler.state_dict()
        save_states = {"num_consumed_batches": num_consumed_batches}
        if only_state_dict:
            driver1.save(Path(path), save_states, dataloader, only_state_dict, should_save_model=True)
        else:
            driver1.save(Path(path), save_states, dataloader, only_state_dict, should_save_model=True, input_spec=[paddle.ones((16, 10))])
        # 加载
        # 更改 batch_size
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=RandomBatchSampler(BatchSampler(dataset, batch_size=2, shuffle=True), 2, False)
        )
        load_states = driver2.load(Path(path), dataloader, only_state_dict, should_load_model=True)
        replaced_loader = load_states.pop("dataloader")
        # 1. 检查 optimizer 的状态
        # TODO optimizer 的 state_dict 总是为空

        # 2. 检查 batch_sampler 是否被正确地加载和替换
        assert not (replaced_loader is dataloader)
        assert replaced_loader.batch_sampler is dataloader.batch_sampler
        assert isinstance(replaced_loader.batch_sampler, RandomBatchSampler)
        assert replaced_loader.batch_sampler.index_list == sampler_states["index_list"]
        assert replaced_loader.batch_sampler.num_consumed_samples == num_consumed_batches * 4

        # 3. 检查 model 的参数是否正确
        # 4. 检查 batch_idx
        start_batch = load_states.pop('batch_idx_in_epoch')
        assert start_batch == 2 * num_consumed_batches
        left_x_batches = set()
        left_y_batches = set()
        for idx, batch in enumerate(replaced_loader):

            left_x_batches.update(batch["x"])
            left_y_batches.update(batch["y"])
            res1 = driver1.model.evaluate_step(**batch)
            res2 = driver2.model.evaluate_step(**batch)
            assert paddle.equal_all(res1["pred"], res2["pred"])

        assert len(left_x_batches) + len(already_seen_x_set) == len(dataset)
        assert len(left_x_batches | already_seen_x_set) == len(dataset)
        assert len(left_y_batches) + len(already_seen_y_set) == len(dataset)
        assert len(left_y_batches | already_seen_y_set) == len(dataset)
    finally:
        synchronize_safe_rm(path)

@pytest.mark.parametrize("only_state_dict", ([True, False]))
def test_save_and_load_with_randomsampler(only_state_dict):
    """
    测试save和load函数，主要测试 dataloader 被替换了 batch_sampler 的情况
    """

    try:
        path = "model.ckp"

        driver1, driver2 = generate_random_driver(10, 10), generate_random_driver(10, 10)
        dataset = PaddleRandomMaxDataset(40, 10)
        batch_sampler = BatchSampler(dataset=dataset, batch_size=4)
        batch_sampler.sampler = RandomSampler(dataset, True)
        dataloader = DataLoader(
            dataset,
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

        sampler_states = dataloader.batch_sampler.sampler.state_dict()
        save_states = {"num_consumed_batches": num_consumed_batches}
        if only_state_dict:
            driver1.save(Path(path), save_states, dataloader, only_state_dict, should_save_model=True)
        else:
            driver1.save(Path(path), save_states, dataloader, only_state_dict, should_save_model=True, input_spec=[paddle.ones((16, 10))])
        
        # 加载
        # 更改 batch_size
        batch_sampler = BatchSampler(dataset=dataset, batch_size=2)
        batch_sampler.sampler = RandomSampler(dataset, True)
        dataloader = DataLoader(
            dataset,
            batch_sampler=batch_sampler
        )
        load_states = driver2.load(Path(path), dataloader, only_state_dict, should_load_model=True)
        replaced_loader = load_states.pop("dataloader")

        # 1. 检查 optimizer 的状态
        # TODO optimizer 的 state_dict 总是为空

        # 2. 检查 sampler 是否被正确地加载和替换
        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
        assert replaced_loader.batch_sampler.sampler.seed == sampler_states["seed"]
        assert replaced_loader.batch_sampler.sampler.epoch == sampler_states["epoch"]
        assert replaced_loader.batch_sampler.sampler.num_consumed_samples == 4 * num_consumed_batches
        assert len(replaced_loader.batch_sampler.sampler.dataset) == sampler_states["length"]
        assert replaced_loader.batch_sampler.sampler.shuffle == sampler_states["shuffle"]

        # 3. 检查 model 的参数是否正确
        # 4. 检查 batch_idx
        start_batch = load_states.pop('batch_idx_in_epoch')
        assert start_batch == 2 * num_consumed_batches
        left_x_batches = set()
        left_y_batches = set()
        for idx, batch in enumerate(replaced_loader):

            left_x_batches.update(batch["x"])
            left_y_batches.update(batch["y"])
            res1 = driver1.model.evaluate_step(**batch)
            res2 = driver2.model.evaluate_step(**batch)
            assert paddle.equal_all(res1["pred"], res2["pred"])

        assert len(left_x_batches) + len(already_seen_x_set) == len(dataset)
        assert len(left_x_batches | already_seen_x_set) == len(dataset)
        assert len(left_y_batches) + len(already_seen_y_set) == len(dataset)
        assert len(left_y_batches | already_seen_y_set) == len(dataset)
    finally:
        synchronize_safe_rm(path)

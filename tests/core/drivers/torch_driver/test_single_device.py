import pytest
from pathlib import Path

from fastNLP.core.drivers.torch_driver.single_device import TorchSingleDriver
from fastNLP.core.samplers import ReproduceBatchSampler, RandomSampler
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1
from tests.helpers.datasets.torch_data import TorchNormalDataset, TorchArgMaxDataset
from tests.helpers.datasets.paddle_data import PaddleNormalDataset
from tests.helpers.models.paddle_model import PaddleNormalModel_Classification_1
from fastNLP.envs.distributed import rank_zero_rm
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE, _NEED_IMPORT_TORCH
if _NEED_IMPORT_TORCH:
    import torch
    from torch.utils.data import DataLoader, BatchSampler
if _NEED_IMPORT_PADDLE:
    import paddle

def dataloader_with_randombatchsampler(dataset, batch_size, shuffle, drop_last):
    """
    建立一个 batch_sampler 为 ReproduceBatchSampler 的 dataloader
    """
    if shuffle:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = DataLoader(
        dataset=dataset,
        batch_sampler=ReproduceBatchSampler(
            BatchSampler(
                sampler, batch_size=batch_size, drop_last=drop_last
            ),
            batch_size=batch_size,
            drop_last=drop_last,
        ),
    )

    return dataloader

def dataloader_with_randomsampler(dataset, batch_size, shuffle, drop_last, seed=0):
    """
    建立一个 sampler 为 RandomSampler 的 dataloader
    """
    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset, shuffle, seed=seed),
        drop_last=drop_last,
        batch_size=batch_size
    )
    return dataloader

############################################################################
#
# 测试基类 TorchDrvier 中的一些简单函数
#
############################################################################

class TestTorchDriverFunctions:
    """
    使用 TorchSingleDriver 测试基类的函数
    """

    @classmethod
    def setup_class(self):
        model = TorchNormalModel_Classification_1(10, 32)
        self.driver = TorchSingleDriver(model, device="cpu")

    @pytest.mark.torchpaddle
    def test_check_single_optimizer_legality(self):
        """
        测试传入单个 optimizer 时的表现
        """
        optimizer = torch.optim.Adam(
            params=self.driver.model.parameters(),
            lr=0.01
        )

        self.driver.set_optimizers(optimizer)

        optimizer = paddle.optimizer.Adam(
            parameters=PaddleNormalModel_Classification_1(10, 32).parameters(),
            learning_rate=0.01,
        )
        # 传入 torch 的 optimize r时，应该报错 ValueError
        with pytest.raises(ValueError):
            self.driver.set_optimizers(optimizer)

    @pytest.mark.torchpaddle
    def test_check_optimizers_legality(self):
        """
        测试传入 optimizer list 的表现
        """
        optimizers = [
            torch.optim.Adam(
                params=self.driver.model.parameters(),
                lr=0.01
            ) for i in range(10)
        ]

        self.driver.set_optimizers(optimizers)

        optimizers += [
            paddle.optimizer.Adam(
                parameters=PaddleNormalModel_Classification_1(10, 32).parameters(),
                learning_rate=0.01,
            )
        ]

        with pytest.raises(ValueError):
            self.driver.set_optimizers(optimizers)

    @pytest.mark.torchpaddle
    def test_check_dataloader_legality_in_train(self):
        """
        测试 `is_train` 参数为 True 时，_check_dataloader_legality 函数的表现
        """
        dataloader = DataLoader(TorchNormalDataset())
        TorchSingleDriver.check_dataloader_legality(dataloader, "dataloader", True)

        # 创建 paddle 的 dataloader
        dataloader = paddle.io.DataLoader(
            PaddleNormalDataset(),
            batch_size=32, shuffle=True
        )
        with pytest.raises(ValueError):
            TorchSingleDriver.check_dataloader_legality(dataloader, "dataloader", True)

    @pytest.mark.torchpaddle
    def test_check_dataloader_legality_in_test(self):
        """
        测试 `is_train` 参数为 False 时，_check_dataloader_legality 函数的表现
        """
        # 此时传入的应该是dict
        dataloader = {
            "train": DataLoader(TorchNormalDataset()),
            "test": DataLoader(TorchNormalDataset())
        }
        TorchSingleDriver.check_dataloader_legality(dataloader, "dataloader", False)

        # 传入的不是 dict，应该报错
        dataloader = DataLoader(TorchNormalDataset())
        with pytest.raises(ValueError):
            TorchSingleDriver.check_dataloader_legality(dataloader, "dataloader", False)

        # 创建 paddle 的 dataloader
        train_loader = paddle.io.DataLoader(
            PaddleNormalDataset(),
            batch_size=32, shuffle=True
        )
        test_loader = paddle.io.DataLoader(
            PaddleNormalDataset(),
            batch_size=32, shuffle=True
        )
        dataloader = {"train": train_loader, "test": test_loader}
        with pytest.raises(ValueError):
            TorchSingleDriver.check_dataloader_legality(dataloader, "dataloader", False)

    @pytest.mark.torch
    def test_tensor_to_numeric(self):
        """
        测试 tensor_to_numeric 函数
        """
        # 单个张量
        tensor = torch.tensor(3)
        res = TorchSingleDriver.tensor_to_numeric(tensor)
        assert res == 3

        tensor = torch.rand((3, 4))
        res = TorchSingleDriver.tensor_to_numeric(tensor)
        assert res == tensor.tolist()

        # 张量list
        tensor_list = [torch.rand((6, 4, 2)) for i in range(10)]
        res = TorchSingleDriver.tensor_to_numeric(tensor_list)
        assert isinstance(res, list)
        tensor_list = [t.tolist() for t in tensor_list]
        assert res == tensor_list

        # 张量tuple
        tensor_tuple = tuple([torch.rand((6, 4, 2)) for i in range(10)])
        res = TorchSingleDriver.tensor_to_numeric(tensor_tuple)
        assert isinstance(res, tuple)
        tensor_tuple = tuple([t.tolist() for t in tensor_tuple])
        assert res == tensor_tuple

        # 张量dict
        tensor_dict = {
            "tensor": torch.rand((3, 4)),
            "list": [torch.rand((6, 4, 2)) for i in range(10)],
            "dict":{
                "list": [torch.rand((6, 4, 2)) for i in range(10)],
                "tensor": torch.rand((3, 4))
            },
            "int": 2,
            "string": "test string"
        }

        res = TorchSingleDriver.tensor_to_numeric(tensor_dict)
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

    @pytest.mark.torch
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

    @pytest.mark.torch
    def test_move_model_to_device_cpu(self):
        """
        测试move_model_to_device函数
        """
        TorchSingleDriver.move_model_to_device(self.driver.model, "cpu")
        assert self.driver.model.linear1.weight.device.type == "cpu"

    @pytest.mark.torch
    def test_move_model_to_device_gpu(self):
        """
        测试move_model_to_device函数
        """
        TorchSingleDriver.move_model_to_device(self.driver.model, "cuda")
        assert self.driver.model.linear1.weight.device.type == "cuda"
        assert self.driver.model.linear1.weight.device.index == 0

    @pytest.mark.torch
    def test_worker_init_function(self):
        """
        测试worker_init_function
        """
        # 先确保不影响运行
        # TODO：正确性
        TorchSingleDriver.worker_init_function(0)

    @pytest.mark.torch
    def test_set_deterministic_dataloader(self):
        """
        测试set_deterministic_dataloader
        """
        # 先确保不影响运行
        # TODO：正确性
        dataloader = DataLoader(TorchNormalDataset())
        self.driver.set_deterministic_dataloader(dataloader)

    @pytest.mark.torch
    def test_set_sampler_epoch(self):
        """
        测试set_sampler_epoch
        """
        # 先确保不影响运行
        # TODO：正确性
        dataloader = DataLoader(TorchNormalDataset())
        self.driver.set_sampler_epoch(dataloader, 0)

    @pytest.mark.torch
    @pytest.mark.parametrize("batch_size", [16])
    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("drop_last", [True, False])
    def test_get_dataloader_args(self, batch_size, shuffle, drop_last):
        """
        测试正常情况下 get_dataloader_args 的表现
        """
        dataloader = DataLoader(
            TorchNormalDataset(),
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        res = TorchSingleDriver.get_dataloader_args(dataloader)

        assert isinstance(res.dataset, TorchNormalDataset)
        assert isinstance(res.batch_sampler, BatchSampler)
        if shuffle:
            assert isinstance(res.sampler, torch.utils.data.RandomSampler)
        else:
            assert isinstance(res.sampler, torch.utils.data.SequentialSampler)
        assert res.shuffle == shuffle
        assert res.batch_size == batch_size
        assert res.drop_last == drop_last

    @pytest.mark.torch
    @pytest.mark.parametrize("batch_size", [16])
    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("drop_last", [True, False])
    def test_get_dataloader_args_with_randombatchsampler(self, batch_size, shuffle, drop_last):
        """
        测试替换了 batch_sampler 后 get_dataloader_args 的表现
        """
        dataset = TorchNormalDataset()
        dataloader = dataloader_with_randombatchsampler(dataset, batch_size, shuffle, drop_last)
        res = TorchSingleDriver.get_dataloader_args(dataloader)

        assert isinstance(res.dataset, TorchNormalDataset)
        assert isinstance(res.batch_sampler, ReproduceBatchSampler)
        if shuffle:
            assert isinstance(res.sampler, torch.utils.data.RandomSampler)
        else:
            assert isinstance(res.sampler, torch.utils.data.SequentialSampler)
        assert res.shuffle == shuffle
        assert res.batch_size == batch_size
        assert res.drop_last == drop_last

    @pytest.mark.torch
    @pytest.mark.parametrize("batch_size", [16])
    @pytest.mark.parametrize("shuffle", [True, False])
    @pytest.mark.parametrize("drop_last", [True, False])
    def test_get_dataloader_args_with_randomsampler(self, batch_size, shuffle, drop_last):
        """
        测试替换了 sampler 后 get_dataloader_args 的表现
        """
        dataset = TorchNormalDataset()
        dataloader = dataloader_with_randomsampler(dataset, batch_size, shuffle, drop_last)
        res = TorchSingleDriver.get_dataloader_args(dataloader)

        assert isinstance(res.dataset, TorchNormalDataset)
        assert isinstance(res.batch_sampler, BatchSampler)
        assert isinstance(res.sampler, RandomSampler)
        assert res.shuffle == shuffle
        assert res.batch_size == batch_size
        assert res.drop_last == drop_last


############################################################################
#
# 测试 TorchSingleDrvier 中的一些简单函数
#
############################################################################

@pytest.mark.torch
class TestSingleDeviceFunction:
    """
    测试其它函数的测试例
    """

    @classmethod
    def setup_class(cls):
        model = TorchNormalModel_Classification_1(10, 784)
        cls.driver = TorchSingleDriver(model, device="cpu")

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
        这个函数仅调用了 torch_move_data_to_device ，测试例在 tests/core/utils/test_torch_utils.py 中
        就不重复测试了
        """
        self.driver.move_data_to_device(torch.rand((32, 64)))


############################################################################
#
# 测试 set_dist_repro_dataloader 函数
#
############################################################################

@pytest.mark.torch
class TestSetDistReproDataloader:
    """
    专门测试 set_dist_repro_dataloader 函数的类
    """
    def setup_method(self):
        self.dataset = TorchNormalDataset(20)
        model = TorchNormalModel_Classification_1(10, 32)
        self.driver = TorchSingleDriver(model, device="cpu")
    
    def test_with_reproducible_false(self):
        """
        测试 set_dist_repro_dataloader 参数 `reproducible` 为 False 时的表现
        当dist为字符串时，此时应该返回原来的 dataloader
        """
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=True)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist="dist", reproducible=False)

        assert replaced_loader is dataloader

    @pytest.mark.parametrize("shuffle", [True, False])
    def test_with_reproducible_true(self, shuffle):
        """
        测试 set_dist_repro_dataloader 参数 `reproducible` 为 True 时的表现
        当dist为字符串时，此时应该返回新的 dataloader，会替换 sampler 为 RandomSampler；
        TODO:
            在 Sampler 的参数不是默认的情况下会替换 batch_sampler 
        """
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=shuffle)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist="dist", reproducible=True)

        assert not (replaced_loader is dataloader)
        # 替换 sampler
        assert isinstance(replaced_loader.batch_sampler, torch.utils.data.BatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert isinstance(replaced_loader.batch_sampler.sampler, RandomSampler)
        assert replaced_loader.batch_sampler.sampler.shuffle == shuffle
        assert replaced_loader.batch_sampler.batch_size == dataloader.batch_sampler.batch_size
        assert replaced_loader.drop_last == dataloader.drop_last

        self.check_set_dist_repro_dataloader(dataloader, replaced_loader, shuffle)

    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_batch_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 参数 dist 不是字符串时的表现，且 dist 是 ReproducibleBatchSampler
        应该返回新的 dataloader，并将 batch_sampler 替换为 dist 对应的 Sampler
        """
        dataloader = DataLoader(self.dataset, batch_size=2, shuffle=shuffle)
        dist = ReproduceBatchSampler(BatchSampler(self.dataset, batch_size=4, drop_last=False), 4, False)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist=dist, reproducible=False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, ReproduceBatchSampler)
        assert replaced_loader.batch_sampler is dist

        self.check_set_dist_repro_dataloader(dataloader, replaced_loader, shuffle)

    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dist_sampler(self, shuffle):
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
    def test_with_dataloader_reproducible_batch_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 参数 dataloader 已经支持断点重训时的表现
        应该返回新的 dataloader，且其余各项设置和原来相同
        """
        dataloader = dataloader_with_randombatchsampler(self.dataset, 4, shuffle, False)
        replaced_loader = self.driver.set_dist_repro_dataloader(dataloader, dist="dist", reproducible=False)

        assert not (replaced_loader is dataloader)
        assert isinstance(replaced_loader.batch_sampler, ReproduceBatchSampler)
        assert not (replaced_loader.batch_sampler is dataloader.batch_sampler)
        assert replaced_loader.batch_sampler.batch_size == dataloader.batch_sampler.batch_size
        assert replaced_loader.drop_last == dataloader.drop_last

        self.check_set_dist_repro_dataloader(dataloader, replaced_loader, shuffle)

    @pytest.mark.parametrize("shuffle", ([True, False]))
    def test_with_dataloader_reproducible_sampler(self, shuffle):
        """
        测试 set_dist_repro_dataloader 参数 dataloader 已经支持断点重训时的表现
        应该返回新的 dataloader，且其余各项设置和原来相同
        """
        dataloader = dataloader_with_randomsampler(self.dataset, 2, shuffle, False)
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
        if isinstance(replaced_loader.batch_sampler, ReproduceBatchSampler):
            sampler_states = replaced_loader.batch_sampler.state_dict()
        else:
            sampler_states = replaced_loader.batch_sampler.sampler.state_dict()

        # 重新加载，应该可以输出剩下的内容，且对于 TorchNormalDataset 来说，排序后应该是一个 range
        left_idxes = set()
        if isinstance(replaced_loader.batch_sampler, ReproduceBatchSampler):
            batch_size = replaced_loader.batch_sampler.batch_size
            sampler_states["num_consumed_samples"] = num_consumed_batches * batch_size
            # 重新改造 dataloader
            new_loader = dataloader_with_randombatchsampler(replaced_loader.dataset, batch_size, shuffle, False)
            new_loader.batch_sampler.load_state_dict(sampler_states)
        else:
            batch_size = replaced_loader.batch_sampler.batch_size
            sampler_states["num_consumed_samples"] = num_consumed_batches * batch_size
            # 重新构造 dataloader
            new_loader = dataloader_with_randomsampler(replaced_loader.dataset, batch_size, shuffle, False)
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

def generate_random_driver(features, labels, fp16=False, device="cpu"):
    """
    生成driver
    """
    model = TorchNormalModel_Classification_1(labels, features)
    opt = torch.optim.Adam(params=model.parameters(), lr=0.01)
    driver = TorchSingleDriver(model, device=device, fp16=fp16)
    driver.set_optimizers(opt)
    driver.setup()

    return driver

@pytest.mark.torch
@pytest.mark.parametrize("only_state_dict", ([True, False]))
def test_save_and_load_model(only_state_dict):
    """
    测试 save_model 和 load_model 函数
    """
    try:
        path = "model"
        dataset = TorchArgMaxDataset(10, 40)
        dataloader = DataLoader(dataset, batch_size=4)
        driver1, driver2 = generate_random_driver(10, 10), generate_random_driver(10, 10)

        driver1.save_model(path, only_state_dict)
        driver2.load_model(path, only_state_dict)

        for batch in dataloader:
            batch = driver1.move_data_to_device(batch)
            res1 = driver1.model.evaluate_step(**batch)
            res2 = driver2.model.evaluate_step(**batch)

            assert torch.equal(res1["preds"], res2["preds"])
    finally:
        rank_zero_rm(path)

@pytest.mark.torch
@pytest.mark.parametrize("only_state_dict", ([True, False]))
@pytest.mark.parametrize("fp16", ([True, False]))
def test_save_and_load_with_randombatchsampler(only_state_dict, fp16):
    """
    测试save和load函数，主要测试 dataloader 被替换了 sampler 之后的情况
    """

    try:
        path = "model.ckp"
        dataset = TorchArgMaxDataset(10, 40)
        dataloader = dataloader_with_randombatchsampler(dataset, 4, True, False)
        driver1, driver2 = generate_random_driver(10, 10, fp16, "cuda"), generate_random_driver(10, 10, False, "cuda")

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
        driver1.save_checkpoint(Path(path), save_states, dataloader, only_state_dict, should_save_model=True)
        # 加载
        # 更改 batch_size

        dataloader = dataloader_with_randombatchsampler(dataset, 2, True, False)
        load_states = driver2.load_checkpoint(Path(path), dataloader, only_state_dict, should_load_model=True)
        replaced_loader = load_states.pop("dataloader")
        # 1. 检查 optimizer 的状态
        # TODO optimizer 的 state_dict 总是为空

        # 2. 检查 batch_sampler 是否被正确地加载和替换
        assert not (replaced_loader is dataloader)
        assert replaced_loader.batch_sampler is dataloader.batch_sampler
        assert isinstance(replaced_loader.batch_sampler, ReproduceBatchSampler)
        assert replaced_loader.batch_sampler.index_list == sampler_states["index_list"]
        assert replaced_loader.batch_sampler.num_consumed_samples == num_consumed_batches * 4

        # 3. 检查 fp16 是否被加载
        if fp16:
            assert isinstance(driver2.grad_scaler, torch.cuda.amp.GradScaler)

        # 4. 检查 model 的参数是否正确
        # 5. 检查 batch_idx
        start_batch = load_states.pop('batch_idx_in_epoch')
        assert start_batch == 2 * num_consumed_batches
        left_x_batches = set()
        left_y_batches = set()
        for idx, batch in enumerate(replaced_loader):

            batch = driver2.move_data_to_device(batch)
            left_x_batches.update(batch["x"])
            left_y_batches.update(batch["y"])
            res1 = driver1.model.evaluate_step(**batch)
            res2 = driver2.model.evaluate_step(**batch)
            assert torch.equal(res1["preds"], res2["preds"])

        assert len(left_x_batches) + len(already_seen_x_set) == len(dataset)
        assert len(left_x_batches | already_seen_x_set) == len(dataset)
        assert len(left_y_batches) + len(already_seen_y_set) == len(dataset)
        assert len(left_y_batches | already_seen_y_set) == len(dataset)
    finally:
        rank_zero_rm(path)

@pytest.mark.torch
@pytest.mark.parametrize("only_state_dict", ([True, False]))
@pytest.mark.parametrize("fp16", ([True, False]))
def test_save_and_load_with_randomsampler(only_state_dict, fp16):
    """
    测试save和load函数，主要测试 dataloader 被替换了 batch_sampler 的情况
    """

    try:
        path = "model.ckp"

        driver1, driver2 = generate_random_driver(10, 10, fp16, "cuda"), generate_random_driver(10, 10, False, "cuda")
        dataset = TorchArgMaxDataset(10, 40)
        dataloader = dataloader_with_randomsampler(dataset, 4, True, False)
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
        driver1.save_checkpoint(Path(path), save_states, dataloader, only_state_dict, should_save_model=True)
        
        # 加载
        # 更改 batch_size
        dataloader = dataloader_with_randomsampler(dataset, 2, True, False)
        load_states = driver2.load_checkpoint(Path(path), dataloader, only_state_dict, should_load_model=True)
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

        # 3. 检查 fp16 是否被加载
        if fp16:
            assert isinstance(driver2.grad_scaler, torch.cuda.amp.GradScaler)

        # 4. 检查 model 的参数是否正确
        # 5. 检查 batch_idx
        start_batch = load_states.pop('batch_idx_in_epoch')
        assert start_batch == 2 * num_consumed_batches
        left_x_batches = set()
        left_y_batches = set()
        for idx, batch in enumerate(replaced_loader):

            batch = driver2.move_data_to_device(batch)
            left_x_batches.update(batch["x"])
            left_y_batches.update(batch["y"])
            res1 = driver1.model.evaluate_step(**batch)
            res2 = driver2.model.evaluate_step(**batch)
            assert torch.equal(res1["preds"], res2["preds"])

        assert len(left_x_batches) + len(already_seen_x_set) == len(dataset)
        assert len(left_x_batches | already_seen_x_set) == len(dataset)
        assert len(left_y_batches) + len(already_seen_y_set) == len(dataset)
        assert len(left_y_batches | already_seen_y_set) == len(dataset)
    finally:
        rank_zero_rm(path)

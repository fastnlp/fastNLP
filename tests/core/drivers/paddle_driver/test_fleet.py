import pytest
import os
import numpy as np
from fastNLP.envs.set_env_on_import import set_env_on_import_paddle

set_env_on_import_paddle()
import paddle
import paddle.distributed as dist
from paddle.io import DataLoader

from fastNLP.core.drivers.paddle_driver.fleet import PaddleFleetDriver
from fastNLP.core.samplers.reproducible_sampler import RandomSampler
from fastNLP.envs import FASTNLP_DISTRIBUTED_CHECK
from tests.helpers.models.paddle_model import PaddleNormalModel_Classification
from tests.helpers.datasets.paddle_data import PaddleDataset_MNIST
from tests.helpers.utils import magic_argv_env_context
from fastNLP.core import synchronize_safe_rm


############################################################################
#
# 测试PaddleFleetDriver的一些函数
#
############################################################################

@magic_argv_env_context
def test_move_data_to_device():
    """
    这个函数仅调用了paddle_move_data_to_device，测试例在tests/core/utils/test_paddle_utils.py中
    就不重复测试了
    """
    try:
        paddle_model = PaddleNormalModel_Classification(10, 784)
        paddle_opt = paddle.optimizer.Adam(parameters=paddle_model.parameters(), learning_rate=0.01)
        driver = PaddleFleetDriver(
            model=paddle_model,
            parallel_device=[0,1],
        )
        driver.set_optimizers(paddle_opt)
        # 区分launch和子进程setup的时候
        if FASTNLP_DISTRIBUTED_CHECK not in os.environ:
            with pytest.raises(SystemExit) as e:
                driver.setup()
            assert e.value.code == 0
            return
        else:
            driver.setup()
        driver.move_data_to_device(paddle.rand((32, 64)))
    finally:
        synchronize_safe_rm("log")

    dist.barrier()


@magic_argv_env_context
def test_is_distributed():
    print(os.getenv("CUDA_VISIBLE_DEVICES"))
    print(paddle.device.get_device())
    try:
        paddle_model = PaddleNormalModel_Classification(10, 784)
        paddle_opt = paddle.optimizer.Adam(parameters=paddle_model.parameters(), learning_rate=0.01)
        driver = PaddleFleetDriver(
            model=paddle_model,
            parallel_device=[0,1],
            output_from_new_proc='all'
        )
        driver.set_optimizers(paddle_opt)
        # 区分launch和子进程setup的时候
        if FASTNLP_DISTRIBUTED_CHECK not in os.environ:
            with pytest.raises(SystemExit) as e:
                driver.setup()
            assert e.value.code == 0
            return
        else:
            driver.setup()
        assert driver.is_distributed() == True
    finally:
        synchronize_safe_rm("log")
    dist.barrier()


@magic_argv_env_context
def test_get_no_sync_context():
    """
    测试能否运行
    """
    try:
        paddle_model = PaddleNormalModel_Classification(10, 784)
        paddle_opt = paddle.optimizer.Adam(parameters=paddle_model.parameters(), learning_rate=0.01)
        driver = PaddleFleetDriver(
            model=paddle_model,
            parallel_device=[0,1],
        )
        driver.set_optimizers(paddle_opt)
        # 区分launch和子进程setup的时候
        if FASTNLP_DISTRIBUTED_CHECK not in os.environ:
            with pytest.raises(SystemExit) as e:
                driver.setup()
            assert e.value.code == 0
            return
        else:
            driver.setup()
        res = driver.get_no_sync_context()
    finally:
        synchronize_safe_rm("log")
    dist.barrier()


@magic_argv_env_context
def test_is_global_zero():
    try:
        paddle_model = PaddleNormalModel_Classification(10, 784)
        paddle_opt = paddle.optimizer.Adam(parameters=paddle_model.parameters(), learning_rate=0.01)
        driver = PaddleFleetDriver(
            model=paddle_model,
            parallel_device=[0,1],
        )
        driver.set_optimizers(paddle_opt)
        # 区分launch和子进程setup的时候
        if FASTNLP_DISTRIBUTED_CHECK not in os.environ:
            with pytest.raises(SystemExit) as e:
                driver.setup()
            assert e.value.code == 0
            return
        else:
            driver.setup()
        driver.is_global_zero()
    finally:
        synchronize_safe_rm("log")
    dist.barrier()



@magic_argv_env_context
def test_unwrap_model():
    try:
        paddle_model = PaddleNormalModel_Classification(10, 784)
        paddle_opt = paddle.optimizer.Adam(parameters=paddle_model.parameters(), learning_rate=0.01)
        driver = PaddleFleetDriver(
            model=paddle_model,
            parallel_device=[0,1],
        )
        driver.set_optimizers(paddle_opt)
        # 区分launch和子进程setup的时候
        if FASTNLP_DISTRIBUTED_CHECK not in os.environ:
            with pytest.raises(SystemExit) as e:
                driver.setup()
            assert e.value.code == 0
            return
        else:
            driver.setup()
        driver.unwrap_model()
    finally:
        synchronize_safe_rm("log")
    dist.barrier()

@magic_argv_env_context
def test_get_local_rank():
    try:
        paddle_model = PaddleNormalModel_Classification(10, 784)
        paddle_opt = paddle.optimizer.Adam(parameters=paddle_model.parameters(), learning_rate=0.01)
        driver = PaddleFleetDriver(
            model=paddle_model,
            parallel_device=[0,1],
        )
        driver.set_optimizers(paddle_opt)
        # 区分launch和子进程setup的时候
        if FASTNLP_DISTRIBUTED_CHECK not in os.environ:
            with pytest.raises(SystemExit) as e:
                driver.setup()
            assert e.value.code == 0
            return
        else:
            driver.setup()
        driver.get_local_rank()
    finally:
        synchronize_safe_rm("log")
    dist.barrier()

@magic_argv_env_context
@pytest.mark.parametrize(
    "dist_sampler",
    ["dist", "unrepeatdist", RandomSampler(PaddleDataset_MNIST("train"))]
)
@pytest.mark.parametrize(
    "reproducible",
    [True, False]
)
def test_replace_sampler(dist_sampler, reproducible):
    """
    测试replace_sampler
    """
    try:
        paddle_model = PaddleNormalModel_Classification(10, 784)
        paddle_opt = paddle.optimizer.Adam(parameters=paddle_model.parameters(), learning_rate=0.01)
        driver = PaddleFleetDriver(
            model=paddle_model,
            parallel_device=[0,1],
        )
        driver.set_optimizers(paddle_opt)
        # 区分launch和子进程setup的时候
        if FASTNLP_DISTRIBUTED_CHECK not in os.environ:
            with pytest.raises(SystemExit) as e:
                driver.setup()
            assert e.value.code == 0
            return
        else:
            driver.setup()
        dataloader = DataLoader(PaddleDataset_MNIST("train"), batch_size=100, shuffle=True)
        driver.set_dist_repro_dataloader(dataloader, dist_sampler, reproducible)
    finally:
        synchronize_safe_rm("log")
    dist.barrier()

############################################################################
#
# 测试单机多卡的训练情况
#
############################################################################

@magic_argv_env_context
class SingleMachineMultiGPUTrainingTestCase:
    """
    测试在单机多卡上使用PaddleFleetDriver进行训练。
    分布式训练用pytest会有些混乱
    """

    def test_case1(self):

        gpus = [0, 1]
        lr = 0.0003
        epochs = 20

        paddle_model = PaddleNormalModel_Classification()

        paddle_opt = paddle.optimizer.Adam(parameters=paddle_model.parameters(), learning_rate=lr)

        train_dataset = PaddleDataset_MNIST("train")
        test_dataset = PaddleDataset_MNIST("test")
        loss_func = paddle.nn.CrossEntropyLoss()

        dataloader = DataLoader(train_dataset, batch_size=100, shuffle=True)

        driver = PaddleFleetDriver(
            model=paddle_model,
            parallel_device=gpus,
        )
        driver.set_optimizers(paddle_opt)
        dataloader = driver.set_dist_repro_dataloader(dataloader, )
        driver.setup()
        # 检查model_device
        self.assertEqual(driver.model_device, f"gpu:{os.environ['PADDLE_LOCAL_DEVICE_IDS']}")

        driver.barrier()

        driver.zero_grad()
        current_epoch_idx = 0
        while current_epoch_idx < epochs:
            epoch_loss, batch = 0, 0
            driver.set_model_mode("train")
            driver.set_sampler_epoch(dataloader, current_epoch_idx)
            for batch, (img, label) in enumerate(dataloader):

                img = paddle.to_tensor(img)
                out = driver.train_step(img)
                label + 1
                loss = loss_func(out, label)
                epoch_loss += loss.item()

                if batch % 50 == 0:
                    print("epoch:{}, batch:{}, loss: {}, rank:{}".format(current_epoch_idx, batch, loss.item(), driver.local_rank))

                driver.backward(loss)
                driver.step()
                driver.zero_grad()
            driver.barrier()
            current_epoch_idx += 1

        # test
        correct = 0
        driver.set_model_mode("eval")
        for img, label in test_dataset:

            img = paddle.to_tensor(np.array(img).astype('float32').reshape(1, -1))
            out = driver.test_step(img)
            res = paddle.nn.functional.softmax(out).argmax().item()
            label = label.item()
            if res == label:
                correct += 1

        print("{} / {}, acc: {}".format(correct, len(test_dataset), correct / len(test_dataset)))

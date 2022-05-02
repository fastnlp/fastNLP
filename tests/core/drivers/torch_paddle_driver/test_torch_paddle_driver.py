import pytest

from fastNLP.modules.mix_modules.mix_module import MixModule
from fastNLP.core.drivers.torch_paddle_driver.torch_paddle_driver import TorchPaddleDriver
from fastNLP.modules.mix_modules.utils import paddle2torch, torch2paddle

import torch
import paddle
from paddle.io import Dataset, DataLoader
import numpy as np

############################################################################
#
# 测试在MNIST数据集上的表现
#
############################################################################

class MNISTDataset(Dataset):
    def __init__(self, dataset):

        self.dataset = [
            (
                np.array(img).astype('float32').reshape(-1),
                label
            ) for img, label in dataset
        ]

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)

class MixMNISTModel(MixModule):
    def __init__(self):
        super(MixMNISTModel, self).__init__()

        self.fc1 = paddle.nn.Linear(784, 64)
        self.fc2 = paddle.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 10)
        self.fc4 = torch.nn.Linear(10, 10)

    def forward(self, x):

        paddle_out = self.fc1(x)
        paddle_out = self.fc2(paddle_out)
        torch_in = paddle2torch(paddle_out)
        torch_out = self.fc3(torch_in)
        torch_out = self.fc4(torch_out)

        return torch_out

    def train_step(self, x):
        return self.forward(x)

    def test_step(self, x):
        return self.forward(x)

@pytest.mark.torchpaddle
class TestMNIST:

    @classmethod
    def setup_class(self):

        self.train_dataset = paddle.vision.datasets.MNIST(mode='train')
        self.test_dataset = paddle.vision.datasets.MNIST(mode='test')
        self.train_dataset = MNISTDataset(self.train_dataset)

        self.lr = 0.0003
        self.epochs = 20

        self.dataloader = DataLoader(self.train_dataset, batch_size=100, shuffle=True)

    def setup_method(self):
        
        model = MixMNISTModel()
        self.torch_loss_func = torch.nn.CrossEntropyLoss()

        torch_opt = torch.optim.Adam(model.parameters(backend="torch"), self.lr)
        paddle_opt = paddle.optimizer.Adam(parameters=model.parameters(backend="paddle"), learning_rate=self.lr)

        self.driver = TorchPaddleDriver(model=model, device="cuda:0")
        self.driver.set_optimizers([torch_opt, paddle_opt])

    def test_case1(self):

        epochs = 20

        self.driver.setup()
        self.driver.zero_grad()
        # 开始训练
        current_epoch_idx = 0
        while current_epoch_idx < epochs:
            epoch_loss, batch = 0, 0
            self.driver.set_model_mode("train")
            self.driver.set_sampler_epoch(self.dataloader, current_epoch_idx)
            for batch, (img, label) in enumerate(self.dataloader):
                img = paddle.to_tensor(img).cuda()
                torch_out = self.driver.train_step(img)
                label = torch.from_numpy(label.numpy()).reshape(-1)
                loss = self.torch_loss_func(torch_out.cpu(), label)
                epoch_loss += loss.item()

                self.driver.backward(loss)
                self.driver.step()
                self.driver.zero_grad()

            current_epoch_idx += 1

        # 开始测试
        correct = 0
        for img, label in self.test_dataset:

            img = paddle.to_tensor(np.array(img).astype('float32').reshape(1, -1))
            torch_out = self.driver.test_step(img)
            res = torch_out.softmax(-1).argmax().item()
            label = label.item()
            if res == label:
                correct += 1

        acc = correct / len(self.test_dataset)
        assert acc > 0.85

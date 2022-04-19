import unittest
import os
from itertools import chain

import torch
import paddle
from paddle.io import Dataset, DataLoader
import numpy as np

from fastNLP.modules.mix_modules.mix_module import MixModule
from fastNLP.modules.mix_modules.utils import paddle2torch, torch2paddle
from fastNLP.core import rank_zero_rm


############################################################################
#
# 测试类的基本功能
#
############################################################################

class TestMixModule(MixModule):
    def __init__(self):
        super(TestMixModule, self).__init__()

        self.torch_fc1 = torch.nn.Linear(10, 10)
        self.torch_softmax = torch.nn.Softmax(0)
        self.torch_conv2d1 = torch.nn.Conv2d(10, 10, 3)
        self.torch_tensor = torch.ones(3, 3)
        self.torch_param = torch.nn.Parameter(torch.ones(4, 4))

        self.paddle_fc1 = paddle.nn.Linear(10, 10)
        self.paddle_softmax = paddle.nn.Softmax(0)
        self.paddle_conv2d1 = paddle.nn.Conv2D(10, 10, 3)
        self.paddle_tensor = paddle.ones((4, 4))

class TestTorchModule(torch.nn.Module):
    def __init__(self):
        super(TestTorchModule, self).__init__()

        self.torch_fc1 = torch.nn.Linear(10, 10)
        self.torch_softmax = torch.nn.Softmax(0)
        self.torch_conv2d1 = torch.nn.Conv2d(10, 10, 3)
        self.torch_tensor = torch.ones(3, 3)
        self.torch_param = torch.nn.Parameter(torch.ones(4, 4))

class TestPaddleModule(paddle.nn.Layer):
    def __init__(self):
        super(TestPaddleModule, self).__init__()

        self.paddle_fc1 = paddle.nn.Linear(10, 10)
        self.paddle_softmax = paddle.nn.Softmax(0)
        self.paddle_conv2d1 = paddle.nn.Conv2D(10, 10, 3)
        self.paddle_tensor = paddle.ones((4, 4))


class TorchPaddleMixModuleTestCase(unittest.TestCase):

    def setUp(self):

        self.model = TestMixModule()
        self.torch_model = TestTorchModule()
        self.paddle_model = TestPaddleModule()

    def test_to(self):
        """
        测试混合模型的to函数
        """
        
        self.model.to("cuda")
        self.torch_model.to("cuda")
        self.paddle_model.to("gpu")
        self.if_device_correct("cuda")

        self.model.to("cuda:2")
        self.torch_model.to("cuda:2")
        self.paddle_model.to("gpu:2")
        self.if_device_correct("cuda:2")

        self.model.to("gpu:1")
        self.torch_model.to("cuda:1")
        self.paddle_model.to("gpu:1")
        self.if_device_correct("cuda:1")

        self.model.to("cpu")
        self.torch_model.to("cpu")
        self.paddle_model.to("cpu")
        self.if_device_correct("cpu")

    def test_train_eval(self):
        """
        测试train和eval函数
        """
        
        self.model.eval()
        self.if_training_correct(False)

        self.model.train()
        self.if_training_correct(True)

    def test_parameters(self):
        """
        测试parameters()函数，由于初始化是随机的，目前仅比较得到结果的长度
        """
        mix_params = []
        params = []

        for value in self.model.named_parameters():
            mix_params.append(value)

        for value in chain(self.torch_model.named_parameters(), self.paddle_model.named_parameters()):
            params.append(value)

        self.assertEqual(len(params), len(mix_params))

    def test_named_parameters(self):
        """
        测试named_parameters函数
        """
        
        mix_param_names = []
        param_names = []

        for name, value in self.model.named_parameters():
            mix_param_names.append(name)

        for name, value in chain(self.torch_model.named_parameters(), self.paddle_model.named_parameters()):
            param_names.append(name)

        self.assertListEqual(sorted(param_names), sorted(mix_param_names))

    def test_torch_named_parameters(self):
        """
        测试对torch参数的提取
        """
        
        mix_param_names = []
        param_names = []

        for name, value in self.model.named_parameters(backend="torch"):
            mix_param_names.append(name)

        for name, value in self.torch_model.named_parameters():
            param_names.append(name)

        self.assertListEqual(sorted(param_names), sorted(mix_param_names))

    def test_paddle_named_parameters(self):
        """
        测试对paddle参数的提取
        """
        
        mix_param_names = []
        param_names = []

        for name, value in self.model.named_parameters(backend="paddle"):
            mix_param_names.append(name)

        for name, value in self.paddle_model.named_parameters():
            param_names.append(name)

        self.assertListEqual(sorted(param_names), sorted(mix_param_names))

    def test_torch_state_dict(self):
        """
        测试提取torch的state dict
        """
        torch_dict = self.torch_model.state_dict()
        mix_dict = self.model.state_dict(backend="torch")

        self.assertListEqual(sorted(torch_dict.keys()), sorted(mix_dict.keys()))

    def test_paddle_state_dict(self):
        """
        测试提取paddle的state dict
        """
        paddle_dict = self.paddle_model.state_dict()
        mix_dict = self.model.state_dict(backend="paddle")

        # TODO 测试程序会显示passed后显示paddle的异常退出信息
        self.assertListEqual(sorted(paddle_dict.keys()), sorted(mix_dict.keys()))

    def test_state_dict(self):
        """
        测试提取所有的state dict
        """
        all_dict = self.torch_model.state_dict()
        all_dict.update(self.paddle_model.state_dict())
        mix_dict = self.model.state_dict()

        # TODO 测试程序会显示passed后显示paddle的异常退出信息
        self.assertListEqual(sorted(all_dict.keys()), sorted(mix_dict.keys()))

    def test_load_state_dict(self):
        """
        测试load_state_dict函数
        """
        state_dict = self.model.state_dict()

        new_model = TestMixModule()
        new_model.load_state_dict(state_dict)
        new_state_dict = new_model.state_dict()

        for name, value in state_dict.items():
            state_dict[name] = value.tolist()
        for name, value in new_state_dict.items():
            new_state_dict[name] = value.tolist()

        self.assertDictEqual(state_dict, new_state_dict)

    def test_save_and_load_state_dict(self):
        """
        测试save_state_dict_to_file和load_state_dict_from_file函数
        """
        path = "model"
        try:
            self.model.save_state_dict_to_file(path)
            new_model = TestMixModule()
            new_model.load_state_dict_from_file(path)

            state_dict = self.model.state_dict()
            new_state_dict = new_model.state_dict()

            for name, value in state_dict.items():
                state_dict[name] = value.tolist()
            for name, value in new_state_dict.items():
                new_state_dict[name] = value.tolist()

            self.assertDictEqual(state_dict, new_state_dict)
        finally:
            rank_zero_rm(path)

    def if_device_correct(self, device):


        self.assertEqual(self.model.torch_fc1.weight.device, self.torch_model.torch_fc1.weight.device)
        self.assertEqual(self.model.torch_conv2d1.weight.device, self.torch_model.torch_fc1.bias.device)
        self.assertEqual(self.model.torch_conv2d1.bias.device, self.torch_model.torch_conv2d1.bias.device)
        self.assertEqual(self.model.torch_tensor.device, self.torch_model.torch_tensor.device)
        self.assertEqual(self.model.torch_param.device, self.torch_model.torch_param.device)

        if device == "cpu":
            self.assertTrue(self.model.paddle_fc1.weight.place.is_cpu_place())
            self.assertTrue(self.model.paddle_fc1.bias.place.is_cpu_place())
            self.assertTrue(self.model.paddle_conv2d1.weight.place.is_cpu_place())
            self.assertTrue(self.model.paddle_conv2d1.bias.place.is_cpu_place())
            self.assertTrue(self.model.paddle_tensor.place.is_cpu_place())
        elif device.startswith("cuda"):
            self.assertTrue(self.model.paddle_fc1.weight.place.is_gpu_place())
            self.assertTrue(self.model.paddle_fc1.bias.place.is_gpu_place())
            self.assertTrue(self.model.paddle_conv2d1.weight.place.is_gpu_place())
            self.assertTrue(self.model.paddle_conv2d1.bias.place.is_gpu_place())
            self.assertTrue(self.model.paddle_tensor.place.is_gpu_place())

            self.assertEqual(self.model.paddle_fc1.weight.place.gpu_device_id(), self.paddle_model.paddle_fc1.weight.place.gpu_device_id())
            self.assertEqual(self.model.paddle_fc1.bias.place.gpu_device_id(), self.paddle_model.paddle_fc1.bias.place.gpu_device_id())
            self.assertEqual(self.model.paddle_conv2d1.weight.place.gpu_device_id(), self.paddle_model.paddle_conv2d1.weight.place.gpu_device_id())
            self.assertEqual(self.model.paddle_conv2d1.bias.place.gpu_device_id(), self.paddle_model.paddle_conv2d1.bias.place.gpu_device_id())
            self.assertEqual(self.model.paddle_tensor.place.gpu_device_id(), self.paddle_model.paddle_tensor.place.gpu_device_id())
        else:
            raise NotImplementedError

    def if_training_correct(self, training):

        self.assertEqual(self.model.torch_fc1.training, training)
        self.assertEqual(self.model.torch_softmax.training, training)
        self.assertEqual(self.model.torch_conv2d1.training, training)

        self.assertEqual(self.model.paddle_fc1.training, training)
        self.assertEqual(self.model.paddle_softmax.training, training)
        self.assertEqual(self.model.paddle_conv2d1.training, training)


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

class TestMNIST(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.train_dataset = paddle.vision.datasets.MNIST(mode='train')
        self.test_dataset = paddle.vision.datasets.MNIST(mode='test')
        self.train_dataset = MNISTDataset(self.train_dataset)

        self.lr = 0.0003
        self.epochs = 20

        self.dataloader = DataLoader(self.train_dataset, batch_size=100, shuffle=True)

    def setUp(self):
        
        self.model = MixMNISTModel().to("cuda")
        self.torch_loss_func = torch.nn.CrossEntropyLoss()

        self.torch_opt = torch.optim.Adam(self.model.parameters(backend="torch"), self.lr)
        self.paddle_opt = paddle.optimizer.Adam(parameters=self.model.parameters(backend="paddle"), learning_rate=self.lr)

    def test_case1(self):

        # 开始训练
        for epoch in range(self.epochs):
            epoch_loss, batch = 0, 0
            for batch, (img, label) in enumerate(self.dataloader):

                img = paddle.to_tensor(img).cuda()
                torch_out = self.model(img)
                label = torch.from_numpy(label.numpy()).reshape(-1)
                loss = self.torch_loss_func(torch_out.cpu(), label)
                epoch_loss += loss.item()

                loss.backward()
                self.torch_opt.step()
                self.paddle_opt.step()
                self.torch_opt.zero_grad()
                self.paddle_opt.clear_grad()

        else:
            self.assertLess(epoch_loss / (batch + 1), 0.3)

        # 开始测试
        correct = 0
        for img, label in self.test_dataset:

            img = paddle.to_tensor(np.array(img).astype('float32').reshape(1, -1))
            torch_out = self.model(img)
            res = torch_out.softmax(-1).argmax().item()
            label = label.item()
            if res == label:
                correct += 1

        acc = correct / len(self.test_dataset)
        self.assertGreater(acc, 0.85)

############################################################################
#
# 测试在ERNIE中文数据集上的表现
#
############################################################################

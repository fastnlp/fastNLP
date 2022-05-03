import pytest
import os

import numpy as np

from fastNLP.core.drivers.jittor_driver.single_device import JittorSingleDriver
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
if _NEED_IMPORT_JITTOR:
    import jittor as jt  # 将 jittor 引入
    from jittor import nn, Module  # 引入相关的模块
    from jittor import init
    from jittor.dataset import MNIST
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Module



class Model(Module):
    def __init__ (self):
        super (Model, self).__init__()
        self.conv1 = nn.Conv (3, 32, 3, 1) # no padding
        
        self.conv2 = nn.Conv (32, 64, 3, 1)
        self.bn = nn.BatchNorm(64)

        self.max_pool = nn.Pool (2, 2)
        self.relu = nn.Relu()
        self.fc1 = nn.Linear (64 * 12 * 12, 256)
        self.fc2 = nn.Linear (256, 10)

    def execute(self, x) : 
        # it's simliar to forward function in Pytorch 
        x = self.conv1 (x)
        x = self.relu (x)
        
        x = self.conv2 (x)
        x = self.bn (x)
        x = self.relu (x)
        
        x = self.max_pool (x)
        x = jt.reshape (x, [x.shape[0], -1])
        x = self.fc1 (x)
        x = self.relu(x)
        x = self.fc2 (x)
        return x

@pytest.mark.jittor
@pytest.mark.skip("Skip jittor tests now.")
class TestSingleDevice:

    def test_on_gpu_without_fp16(self):
        # TODO get_dataloader
        batch_size = 64
        learning_rate = 0.1
        epochs = 5
        losses = []
        losses_idx = []

        train_loader = MNIST(train=True, batch_size=batch_size, shuffle=True)
        val_loader = MNIST(train=False, batch_size=1, shuffle=False)

        model = Model()
        driver = JittorSingleDriver(model, device=[1])
        optimizer = nn.SGD(model.parameters(), learning_rate)
        driver.set_optimizers(optimizer)

        for epoch in range(epochs):
            driver.set_model_mode("train")
            lens = len(train_loader)
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                outputs =driver.train_step(inputs)
                loss = nn.cross_entropy_loss(outputs, targets)
                driver.backward(loss)
                driver.step()
                driver.zero_grad()
                losses.append(loss.data[0])
                losses_idx.append(epoch * lens + batch_idx)

        test_loss = 0
        correct = 0
        total_acc = 0
        total_num = 0
        driver.set_model_mode("eval")
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            batch_size = inputs.shape[0]
            outputs = driver.test_step(inputs)
            pred = np.argmax(outputs.data, axis=1)
            acc = np.sum(targets.data==pred)
            total_acc += acc
            total_num += batch_size
            acc = acc / batch_size  	
        assert total_acc / total_num > 0.95


    def test_on_cpu_without_fp16(self):
        pass

    def test_on_gpu_with_fp16(self):
        pass
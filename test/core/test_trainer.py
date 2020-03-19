import time
import unittest

import numpy as np
import torch.nn.functional as F
from torch import nn

from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import BCELoss
from fastNLP import CrossEntropyLoss
from fastNLP import AccuracyMetric
from fastNLP import SGD
from fastNLP import Trainer
from fastNLP.models.base_model import NaiveClassifier

def prepare_fake_dataset():
    mean = np.array([-3, -3])
    cov = np.array([[1, 0], [0, 1]])
    class_A = np.random.multivariate_normal(mean, cov, size=(1000,))
    
    mean = np.array([3, 3])
    cov = np.array([[1, 0], [0, 1]])
    class_B = np.random.multivariate_normal(mean, cov, size=(1000,))
    
    data_set = DataSet([Instance(x=[float(item[0]), float(item[1])], y=[0.0]) for item in class_A] +
                       [Instance(x=[float(item[0]), float(item[1])], y=[1.0]) for item in class_B])
    return data_set


def prepare_fake_dataset2(*args, size=100):
    ys = np.random.randint(4, size=100, dtype=np.int64)
    data = {'y': ys}
    for arg in args:
        data[arg] = np.random.randn(size, 5)
    return DataSet(data=data)


class TrainerTestGround(unittest.TestCase):
    def test_case(self):
        data_set = prepare_fake_dataset()
        data_set.set_input("x", flag=True)
        data_set.set_target("y", flag=True)
        
        train_set, dev_set = data_set.split(0.3)
        
        model = NaiveClassifier(2, 1)
        
        trainer = Trainer(train_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=10, print_every=50, dev_data=dev_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), validate_every=-1, save_path=None,
                          use_tqdm=True, check_code_level=2)
        trainer.train()
        """
        # 应该正确运行
        """

    def test_save_path(self):
        data_set = prepare_fake_dataset()
        data_set.set_input("x", flag=True)
        data_set.set_target("y", flag=True)

        train_set, dev_set = data_set.split(0.3)

        model = NaiveClassifier(2, 1)

        save_path = 'test_save_models'

        trainer = Trainer(train_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=10, print_every=50, dev_data=dev_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), validate_every=-1, save_path=save_path,
                          use_tqdm=True, check_code_level=2)
        trainer.train()

    
    def test_trainer_suggestion1(self):
        # 检查报错提示能否正确提醒用户。
        # 这里没有传入forward需要的数据。需要trainer提醒用户如何设置。
        dataset = prepare_fake_dataset2('x')
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 4)
            
            def forward(self, x1, x2, y):
                x1 = self.fc(x1)
                x2 = self.fc(x2)
                x = x1 + x2
                loss = F.cross_entropy(x, y)
                return {'loss': loss}
        
        model = Model()
        
        with self.assertRaises(RuntimeError):
            trainer = Trainer(train_data=dataset, model=model)
        """
        # 应该获取到的报错提示
        NameError: 
        The following problems occurred when calling Model.forward(self, x1, x2, y)
        missing param: ['y', 'x1', 'x2']
        Suggestion: (1). You might need to set ['y'] as input. 
                    (2). You need to provide ['x1', 'x2'] in DataSet and set it as input. 

        """
    
    def test_trainer_suggestion2(self):
        # 检查报错提示能否正确提醒用户
        # 这里传入forward需要的数据，看是否可以运行
        dataset = prepare_fake_dataset2('x1', 'x2')
        dataset.set_input('x1', 'x2', 'y', flag=True)
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 4)
            
            def forward(self, x1, x2, y):
                x1 = self.fc(x1)
                x2 = self.fc(x2)
                x = x1 + x2
                loss = F.cross_entropy(x, y)
                return {'loss': loss}
        
        model = Model()
        trainer = Trainer(train_data=dataset, model=model, print_every=2, use_tqdm=False)
        trainer.train()
        """
        # 应该正确运行
        """
    
    def test_trainer_suggestion3(self):
        # 检查报错提示能否正确提醒用户
        # 这里传入forward需要的数据，但是forward没有返回loss这个key
        dataset = prepare_fake_dataset2('x1', 'x2')
        dataset.set_input('x1', 'x2', 'y', flag=True)
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 4)
            
            def forward(self, x1, x2, y):
                x1 = self.fc(x1)
                x2 = self.fc(x2)
                x = x1 + x2
                loss = F.cross_entropy(x, y)
                return {'wrong_loss_key': loss}
        
        model = Model()
        with self.assertRaises(NameError):
            trainer = Trainer(train_data=dataset, model=model, print_every=2, use_tqdm=False)
            trainer.train()
    
    def test_trainer_suggestion4(self):
        # 检查报错提示能否正确提醒用户
        # 这里传入forward需要的数据，是否可以正确提示unused
        dataset = prepare_fake_dataset2('x1', 'x2')
        dataset.set_input('x1', 'x2', 'y', flag=True)
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 4)
            
            def forward(self, x1, x2, y):
                x1 = self.fc(x1)
                x2 = self.fc(x2)
                x = x1 + x2
                loss = F.cross_entropy(x, y)
                return {'losses': loss}
        
        model = Model()
        with self.assertRaises(NameError):
            trainer = Trainer(train_data=dataset, model=model, print_every=2, use_tqdm=False)
    
    def test_trainer_suggestion5(self):
        # 检查报错提示能否正确提醒用户
        # 这里传入多余参数，让其duplicate, 但这里因为y不会被调用，所以其实不会报错
        dataset = prepare_fake_dataset2('x1', 'x_unused')
        dataset.rename_field('x_unused', 'x2')
        dataset.set_input('x1', 'x2', 'y')
        dataset.set_target('y')
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 4)
            
            def forward(self, x1, x2, y):
                x1 = self.fc(x1)
                x2 = self.fc(x2)
                x = x1 + x2
                loss = F.cross_entropy(x, y)
                return {'loss': loss}
        
        model = Model()
        trainer = Trainer(train_data=dataset, model=model, print_every=2, use_tqdm=False)
    
    def test_trainer_suggestion6(self):
        # 检查报错提示能否正确提醒用户
        # 这里传入多余参数，让其duplicate
        dataset = prepare_fake_dataset2('x1', 'x_unused')
        dataset.rename_field('x_unused', 'x2')
        dataset.set_input('x1', 'x2')
        dataset.set_target('y', 'x1')
        
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 4)
            
            def forward(self, x1, x2):
                x1 = self.fc(x1)
                x2 = self.fc(x2)
                x = x1 + x2
                time.sleep(0.1)
                # loss = F.cross_entropy(x, y)
                return {'preds': x}
        
        model = Model()
        with self.assertRaises(NameError):
            trainer = Trainer(train_data=dataset, model=model, loss=CrossEntropyLoss(), print_every=2, dev_data=dataset,
                              metrics=AccuracyMetric(), use_tqdm=False)
    
    """
    def test_trainer_multiprocess(self):
        dataset = prepare_fake_dataset2('x1', 'x2')
        dataset.set_input('x1', 'x2', 'y', flag=True)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 4)

            def forward(self, x1, x2, y):
                x1 = self.fc(x1)
                x2 = self.fc(x2)
                x = x1 + x2
                loss = F.cross_entropy(x, y)
                return {'loss': loss}

        model = Model()
        trainer = Trainer(
            train_data=dataset,
            model=model,
            use_tqdm=True,
            print_every=2,
            num_workers=2,
            pin_memory=False,
            timeout=0,
        )
        trainer.train()
    """

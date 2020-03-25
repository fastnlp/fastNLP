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
from fastNLP import TorchLoaderIter


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
        import os
        if os.path.exists(save_path):
            import shutil
            shutil.rmtree(save_path)

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

    def test_udf_dataiter(self):
        import random
        import torch
        class UdfDataSet:
            def __init__(self, num_samples):
                self.num_samples = num_samples

            def __getitem__(self, idx):
                x = [random.random() for _ in range(3)]
                y = random.random()
                return x,y

            def __len__(self):
                return self.num_samples

        def collect_fn(data_list):
            # [(x1,y1), (x2,y2), ...], 这里的输入实际上是将UdfDataSet的__getitem__输入结合为list
            xs, ys = [], []
            for l in data_list:
                x, y = l
                xs.append(x)
                ys.append(y)
            x,y = torch.FloatTensor(xs), torch.FloatTensor(ys)
            return {'x':x, 'y':y}, {'y':y}

        dataset = UdfDataSet(10)
        dataset = TorchLoaderIter(dataset, collate_fn=collect_fn)
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3, 1)
            def forward(self, x, y):
                return {'loss':torch.pow(self.fc(x).squeeze(-1)-y, 2).sum()}
            def predict(self, x):
                return {'pred':self.fc(x).squeeze(0)}
        model = Model()
        trainer = Trainer(train_data=dataset, model=model, loss=None, print_every=2, dev_data=dataset,
                          metrics=AccuracyMetric(target='y'), use_tqdm=False)
        trainer.train(load_best_model=False)

    def test_onthefly_iter(self):
        import tempfile
        import random
        import torch
        tmp_file_handler, tmp_file_path = tempfile.mkstemp(text=True)
        try:
            num_samples = 10
            data = []
            for _ in range(num_samples):
                x, y = [random.random() for _ in range(3)], random.random()
                data.append(x + [y])
            with open(tmp_file_path, 'w') as f:
                for d in data:
                    f.write(' '.join(map(str, d)) + '\n')

            class FileDataSet:
                def __init__(self, tmp_file):
                    num_samples = 0
                    line_pos = [0]  # 对应idx是某一行对应的位置
                    self.tmp_file_handler = open(tmp_file, 'r', encoding='utf-8')
                    line = self.tmp_file_handler.readline()
                    while line:
                        if line.strip():
                            num_samples += 1
                            line_pos.append(self.tmp_file_handler.tell())
                        line = self.tmp_file_handler.readline()
                    self.tmp_file_handler.seek(0)
                    self.num_samples = num_samples
                    self.line_pos = line_pos

                def __getitem__(self, idx):
                    line_start, line_end = self.line_pos[idx], self.line_pos[idx + 1]
                    self.tmp_file_handler.seek(line_start)
                    line = self.tmp_file_handler.read(line_end - line_start).strip()
                    values = list(map(float, line.split()))
                    gold_d = data[idx]
                    assert all([g==v for g,v in zip(gold_d, values)]), "Should have the same data"
                    x, y = values[:3], values[-1]
                    return x, y

                def __len__(self):
                    return self.num_samples

            def collact_fn(data_list):
                # [(x1,y1), (x2,y2), ...], 这里的输入实际上是将UdfDataSet的__getitem__输入结合为list
                xs, ys = [], []
                for l in data_list:
                    x, y = l
                    xs.append(x)
                    ys.append(y)
                x, y = torch.FloatTensor(xs), torch.FloatTensor(ys)
                return {'x': x, 'y': y}, {'y': y}

            dataset = FileDataSet(tmp_file_path)
            dataset = TorchLoaderIter(dataset, collate_fn=collact_fn)

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(3, 1)

                def forward(self, x, y):
                    return {'loss': torch.pow(self.fc(x).squeeze(-1) - y, 2).sum()}

                def predict(self, x):
                    return {'pred': self.fc(x).squeeze(0)}

            model = Model()
            trainer = Trainer(train_data=dataset, model=model, loss=None, print_every=2, dev_data=dataset,
                              metrics=AccuracyMetric(target='y'), use_tqdm=False, n_epochs=2)
            trainer.train(load_best_model=False)

        finally:
            import os
            if os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    def test_collecct_fn(self):
        dataset = prepare_fake_dataset2('x1', 'x2')
        dataset.set_input('x1', 'x2')
        dataset.set_target('y', 'x1')
        import torch
        def fn(ins_list):
            x = []
            for ind, ins in ins_list:
                x.append(ins['x1']+ins['x2'])
            x = torch.FloatTensor(x)
            return {'x':x}, {}
        dataset.add_collect_fn(fn)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 4)

            def forward(self, x1, x2, x):
                x1 = self.fc(x1)
                x2 = self.fc(x2)
                x = self.fc(x)
                sum_x = x1 + x2 + x
                time.sleep(0.1)
                # loss = F.cross_entropy(x, y)
                return {'pred': sum_x}

        model = Model()
        trainer = Trainer(train_data=dataset, model=model, loss=CrossEntropyLoss(target='y'), print_every=2,
                          dev_data=dataset, metrics=AccuracyMetric(target='y'), use_tqdm=False)
        trainer.train()

    def test_collect_fn2(self):
        """测试能否实现batch_x, batch_y"""
        dataset = prepare_fake_dataset2('x1', 'x2')
        dataset.set_input('x1', 'x2')
        dataset.set_target('y', 'x1')
        import torch
        def fn(ins_list):
            x = []
            for ind, ins in ins_list:
                x.append(ins['x1']+ins['x2'])
            x = torch.FloatTensor(x)
            return {'x':x}, {'target':x[:, :4].argmax(dim=-1)}
        dataset.add_collect_fn(fn)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 4)

            def forward(self, x1, x2, x):
                x1 = self.fc(x1)
                x2 = self.fc(x2)
                x = self.fc(x)
                sum_x = x1 + x2 + x
                time.sleep(0.1)
                # loss = F.cross_entropy(x, y)
                return {'pred': sum_x}

        model = Model()
        trainer = Trainer(train_data=dataset, model=model, loss=CrossEntropyLoss(), print_every=2,
                          dev_data=dataset, metrics=AccuracyMetric(), use_tqdm=False)
        trainer.train()

    def test_collect_fn3(self):
        """
        测试应该会覆盖

        :return:
        """
        dataset = prepare_fake_dataset2('x1', 'x2')
        dataset.set_input('x1', 'x2')
        dataset.set_target('y')
        import torch
        def fn(ins_list):
            x = []
            for ind, ins in ins_list:
                x.append(ins['x1']+ins['x2'])
            x = torch.FloatTensor(x)
            return {'x1':torch.zeros_like(x)}, {'target':torch.zeros(x.size(0)).long(), 'y':x}
        dataset.add_collect_fn(fn)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(5, 1, bias=False)

            def forward(self, x1):
                x1 = self.fc(x1)
                assert x1.sum()==0, "Should be replaced to one"
                # loss = F.cross_entropy(x, y)
                return {'pred': x1}

        model = Model()
        trainer = Trainer(train_data=dataset, model=model, loss=CrossEntropyLoss(), print_every=2,
                          dev_data=dataset, metrics=AccuracyMetric(), use_tqdm=False, n_epochs=1)
        best_metric = trainer.train()['best_eval']['AccuracyMetric']['acc']
        self.assertTrue(best_metric==1)

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

import unittest

import numpy as np
import torch
import os
import shutil

from fastNLP.core.callback import EarlyStopCallback, GradientClipCallback, LRScheduler, ControlC, \
    LRFinder, TensorboardCallback
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import BCELoss
from fastNLP import AccuracyMetric
from fastNLP import SGD
from fastNLP import Trainer
from fastNLP.models.base_model import NaiveClassifier
from fastNLP.core.callback import EarlyStopError
from fastNLP.core.callback import EvaluateCallback, FitlogCallback, SaveModelCallback
from fastNLP.core.callback import WarmupCallback

def prepare_env():
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
    
    data_set = prepare_fake_dataset()
    data_set.set_input("x")
    data_set.set_target("y")
    model = NaiveClassifier(2, 1)
    return data_set, model


class TestCallback(unittest.TestCase):
    
    def test_gradient_clip(self):
        data_set, model = prepare_env()
        trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=20, print_every=50, dev_data=data_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=False,
                          callbacks=[GradientClipCallback(model.parameters(), clip_value=2)], check_code_level=2)
        trainer.train()
    
    def test_early_stop(self):
        data_set, model = prepare_env()
        trainer = Trainer(data_set, model, optimizer=SGD(lr=0.01), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=20, print_every=50, dev_data=data_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=False,
                          callbacks=[EarlyStopCallback(5)], check_code_level=2)
        trainer.train()
    
    def test_lr_scheduler(self):
        data_set, model = prepare_env()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = Trainer(data_set, model, optimizer=optimizer, loss=BCELoss(pred="predict", target="y"), batch_size=32,
                          n_epochs=5, print_every=50, dev_data=data_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=False,
                          callbacks=[LRScheduler(torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1))],
                          check_code_level=2)
        trainer.train()
    
    def test_KeyBoardInterrupt(self):
        data_set, model = prepare_env()
        trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=5, print_every=50, use_tqdm=False, callbacks=[ControlC(False)],
                          check_code_level=2)
        trainer.train()
    
    def test_LRFinder(self):
        data_set, model = prepare_env()
        trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=5, print_every=50, use_tqdm=False,
                          callbacks=[LRFinder(len(data_set) // 32)], check_code_level=2)
        trainer.train()
    
    def test_TensorboardCallback(self):
        data_set, model = prepare_env()
        trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=5, print_every=50, dev_data=data_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=False,
                          callbacks=[TensorboardCallback("loss", "metric")], check_code_level=2)
        trainer.train()
        import os
        import shutil
        path = os.path.join("./", 'tensorboard_logs_{}'.format(trainer.start_time))
        if os.path.exists(path):
            shutil.rmtree(path)
    
    def test_readonly_property(self):
        from fastNLP.core.callback import Callback
        passed_epochs = []
        total_epochs = 5
        
        class MyCallback(Callback):
            def __init__(self):
                super(MyCallback, self).__init__()
            
            def on_epoch_begin(self):
                passed_epochs.append(self.epoch)
                print(self.n_epochs, self.n_steps, self.batch_size)
                print(self.model)
                print(self.optimizer)
        
        data_set, model = prepare_env()
        trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=total_epochs, print_every=50, dev_data=data_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=False, callbacks=[MyCallback()],
                          check_code_level=2)
        trainer.train()
        assert passed_epochs == list(range(1, total_epochs + 1))

    def test_evaluate_callback(self):
        data_set, model = prepare_env()
        from fastNLP import Tester
        tester = Tester(data=data_set, model=model, metrics=AccuracyMetric(pred="predict", target="y"))
        evaluate_callback = EvaluateCallback(data_set, tester)

        trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=5, print_every=50, dev_data=data_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=False,
                          callbacks=evaluate_callback, check_code_level=2)
        trainer.train()

    def test_fitlog_callback(self):
        import fitlog
        os.makedirs('logs/')
        fitlog.set_log_dir('logs/')
        data_set, model = prepare_env()
        from fastNLP import Tester
        tester = Tester(data=data_set, model=model, metrics=AccuracyMetric(pred="predict", target="y"))
        fitlog_callback = FitlogCallback(data_set, tester)

        trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=5, print_every=50, dev_data=data_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=True,
                          callbacks=fitlog_callback, check_code_level=2)
        trainer.train()
        shutil.rmtree('logs/')

    def test_save_model_callback(self):
        data_set, model = prepare_env()
        top = 3
        save_model_callback = SaveModelCallback('save_models/', top=top)
        trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=5, print_every=50, dev_data=data_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=True,
                          callbacks=save_model_callback, check_code_level=2)
        trainer.train()

        timestamp = os.listdir('save_models')[0]
        self.assertEqual(len(os.listdir(os.path.join('save_models', timestamp))), top)
        shutil.rmtree('save_models/')

    def test_warmup_callback(self):
        data_set, model = prepare_env()
        warmup_callback = WarmupCallback()
        trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=5, print_every=50, dev_data=data_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=True,
                          callbacks=warmup_callback, check_code_level=2)
        trainer.train()

import os
import tempfile
import unittest

import numpy as np
import torch

from fastNLP import AccuracyMetric
from fastNLP import BCELoss
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import SGD
from fastNLP import Trainer
from fastNLP.core.callback import EarlyStopCallback, GradientClipCallback, LRScheduler, ControlC, \
    LRFinder, TensorboardCallback
from fastNLP.core.callback import EvaluateCallback, FitlogCallback, SaveModelCallback
from fastNLP.core.callback import WarmupCallback
from fastNLP.models.base_model import NaiveClassifier


def prepare_env():
    mean = np.array([-3, -3])
    cov = np.array([[1, 0], [0, 1]])
    class_A = np.random.multivariate_normal(mean, cov, size=(1000,))
    
    mean = np.array([3, 3])
    cov = np.array([[1, 0], [0, 1]])
    class_B = np.random.multivariate_normal(mean, cov, size=(1000,))
    
    data_set = DataSet([Instance(x=[float(item[0]), float(item[1])], y=[0.0]) for item in class_A] +
                       [Instance(x=[float(item[0]), float(item[1])], y=[1.0]) for item in class_B])
    
    data_set.set_input("x")
    data_set.set_target("y")
    model = NaiveClassifier(2, 1)
    return data_set, model


class TestCallback(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
    
    def tearDown(self):
        import shutil
        shutil.rmtree(self.tempdir)
    
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
        fitlog.set_log_dir(self.tempdir, new_log=True)
        data_set, model = prepare_env()
        from fastNLP import Tester
        tester = Tester(data=data_set, model=model, metrics=AccuracyMetric(pred="predict", target="y"))
        fitlog_callback = FitlogCallback(data_set, tester)
        
        trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=5, print_every=50, dev_data=data_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=True,
                          callbacks=fitlog_callback, check_code_level=2)
        trainer.train()

    def test_CheckPointCallback(self):

        from fastNLP import CheckPointCallback, Callback
        from fastNLP import Tester

        class RaiseCallback(Callback):
            def __init__(self, stop_step=10):
                super().__init__()
                self.stop_step = stop_step

            def on_backward_begin(self, loss):
                if self.step > self.stop_step:
                    raise RuntimeError()

        data_set, model = prepare_env()
        tester = Tester(data=data_set, model=model, metrics=AccuracyMetric(pred="predict", target="y"))
        import fitlog

        fitlog.set_log_dir(self.tempdir, new_log=True)
        tempfile_path = os.path.join(self.tempdir, 'chkt.pt')
        callbacks = [CheckPointCallback(tempfile_path)]

        fitlog_callback = FitlogCallback(data_set, tester)
        callbacks.append(fitlog_callback)

        callbacks.append(RaiseCallback(100))
        try:
            trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                              batch_size=32, n_epochs=5, print_every=50, dev_data=data_set,
                              metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=True,
                              callbacks=callbacks, check_code_level=2)
            trainer.train()
        except:
            pass
        #  用下面的代码模拟重新运行
        data_set, model = prepare_env()
        callbacks = [CheckPointCallback(tempfile_path)]
        tester = Tester(data=data_set, model=model, metrics=AccuracyMetric(pred="predict", target="y"))
        fitlog_callback = FitlogCallback(data_set, tester)
        callbacks.append(fitlog_callback)

        trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=5, print_every=50, dev_data=data_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=True,
                          callbacks=callbacks, check_code_level=2)
        trainer.train()

    def test_save_model_callback(self):
        data_set, model = prepare_env()
        top = 3
        save_model_callback = SaveModelCallback(self.tempdir, top=top)
        trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=5, print_every=50, dev_data=data_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=True,
                          callbacks=save_model_callback, check_code_level=2)
        trainer.train()
        
        timestamp = os.listdir(self.tempdir)[0]
        self.assertEqual(len(os.listdir(os.path.join(self.tempdir, timestamp))), top)
    
    def test_warmup_callback(self):
        data_set, model = prepare_env()
        warmup_callback = WarmupCallback()
        trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=32, n_epochs=5, print_every=50, dev_data=data_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=True,
                          callbacks=warmup_callback, check_code_level=2)
        trainer.train()
    
    def test_early_stop_callback(self):
        """
        需要观察是否真的 EarlyStop
        """
        data_set, model = prepare_env()
        trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                          batch_size=2, n_epochs=10, print_every=5, dev_data=data_set,
                          metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=True,
                          callbacks=EarlyStopCallback(1), check_code_level=2)
        trainer.train()

@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
def test_control_C():
    # 用于测试 ControlC , 再两次训练时用 Control+C 进行退出，如果最后不显示 "Test failed!" 则通过测试
    from fastNLP import ControlC, Callback
    import time

    line1 = "\n\n\n\n\n*************************"
    line2 = "*************************\n\n\n\n\n"

    class Wait(Callback):
        def on_epoch_end(self):
            time.sleep(5)

    data_set, model = prepare_env()

    print(line1 + "Test starts!" + line2)
    trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                      batch_size=32, n_epochs=20, dev_data=data_set,
                      metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=True,
                      callbacks=[Wait(), ControlC(False)], check_code_level=2)
    trainer.train()

    print(line1 + "Program goes on ..." + line2)

    trainer = Trainer(data_set, model, optimizer=SGD(lr=0.1), loss=BCELoss(pred="predict", target="y"),
                      batch_size=32, n_epochs=20, dev_data=data_set,
                      metrics=AccuracyMetric(pred="predict", target="y"), use_tqdm=True,
                      callbacks=[Wait(), ControlC(True)], check_code_level=2)
    trainer.train()

    print(line1 + "Test failed!" + line2)


if __name__ == "__main__":
    test_control_C()

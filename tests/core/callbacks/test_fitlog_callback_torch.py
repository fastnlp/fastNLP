import os.path
import shutil

import pytest

from fastNLP import FitlogCallback, Metric
from fastNLP.core.callbacks.checkpoint_callback import CheckpointCallback
from fastNLP.core.controllers.trainer import Trainer
from fastNLP.envs import _module_available
from fastNLP.envs.distributed import rank_zero_rm
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from tests.helpers.datasets.torch_data import TorchArgMaxDataset
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1

if _NEED_IMPORT_TORCH:
    from torch.optim import SGD
    from torch.utils.data import DataLoader


class DemoMetric(Metric):
    """不断下降的metric."""

    def __init__(self):
        super(DemoMetric, self).__init__()
        self.count = 100

    def update(self, **kwargs):
        pass

    def get_metric(self) -> dict:
        self.count -= 1
        return {'acc': self.count}


@pytest.mark.torch
@pytest.mark.skipif(not _module_available('fitlog'), reason='no fitlog')
def test_fitlog_callback_rerun():
    # 正常运行的fitlog测试
    tmp_log_dir = 'tmp_logs'
    ckpt_save_folder = 'ckpt_save_folder'
    try:
        import fitlog
        os.mkdir(tmp_log_dir)
        fitlog.set_log_dir(tmp_log_dir)

        fitlog.add_hyper(1e-3, name='lr')
        clb = FitlogCallback(monitor='acc')
        clb2 = CheckpointCallback(
            folder=ckpt_save_folder,
            topk=1,
            save_object='trainer',
            monitor='acc')
        ds = TorchArgMaxDataset(data_num=20, feature_dimension=2)
        dl = DataLoader(dataset=ds, batch_size=10, shuffle=True)

        model = TorchNormalModel_Classification_1(2, feature_dimension=2)
        optimizer = SGD(model.parameters(), lr=1e-3)

        trainer = Trainer(
            model,
            train_dataloader=dl,
            optimizers=optimizer,
            device='cpu',
            n_epochs=5,
            evaluate_dataloaders=dl,
            callbacks=[clb, clb2],
            metrics={'acc': DemoMetric()})
        trainer.run()
        assert len(os.listdir(tmp_log_dir)) == 1  # 应该生成了一个文件了

        exact_log_dir = os.listdir(tmp_log_dir)[0]
        with open(os.path.join(tmp_log_dir, exact_log_dir,
                               'best_metric.log')) as f:
            num_line_in_best_metric = len(f.readlines())
        # fitlog.set_log_dir(os.path.join(tmp_log_dir, exact_log_dir))

        clb = FitlogCallback(monitor='acc')
        trainer = Trainer(
            model,
            train_dataloader=dl,
            optimizers=optimizer,
            device='cpu',
            n_epochs=10,
            evaluate_dataloaders=dl,
            callbacks=[clb],
            metrics={'acc': DemoMetric()})
        ckpt_save_folder1 = os.listdir(ckpt_save_folder)[0]
        ckpt_save_folder2 = os.listdir(
            os.path.join(ckpt_save_folder, ckpt_save_folder1))[0]
        trainer.load_checkpoint(
            os.path.join(ckpt_save_folder, ckpt_save_folder1,
                         ckpt_save_folder2))
        trainer.run()

        assert len(os.listdir(tmp_log_dir)) == 1  # 应该还是只有一个文件夹
        with open(os.path.join(tmp_log_dir, exact_log_dir,
                               'best_metric.log')) as f:
            new_num_line_in_best_metric = len(f.readlines())
        assert new_num_line_in_best_metric == num_line_in_best_metric

    finally:
        rank_zero_rm(tmp_log_dir)
        rank_zero_rm(ckpt_save_folder)


@pytest.mark.torch
@pytest.mark.skipif(not _module_available('fitlog'), reason='no fitlog')
def test_fitlog_callback():
    # 正常运行的fitlog测试
    tmp_log_dir = 'tmp_logs'
    try:
        import fitlog
        os.mkdir(tmp_log_dir)
        fitlog.set_log_dir(tmp_log_dir)

        fitlog.add_hyper(1e-3, name='lr')
        clb = FitlogCallback(monitor='acc')
        ds = TorchArgMaxDataset(data_num=20, feature_dimension=2)
        dl = DataLoader(dataset=ds, batch_size=10, shuffle=True)

        model = TorchNormalModel_Classification_1(2, feature_dimension=2)
        optimizer = SGD(model.parameters(), lr=1e-3)

        trainer = Trainer(
            model,
            train_dataloader=dl,
            optimizers=optimizer,
            device='cpu',
            n_epochs=5,
            evaluate_dataloaders=dl,
            callbacks=[clb],
            metrics={'acc': DemoMetric()})
        trainer.run()
        assert len(os.listdir(tmp_log_dir)) == 1  # 应该生成了一个文件了
    finally:
        if os.path.exists(tmp_log_dir) and os.path.isdir(tmp_log_dir):
            shutil.rmtree(tmp_log_dir)

import pytest
from fastNLP.core.callbacks import callback

from fastNLP.core.controllers.trainer import Trainer
from fastNLP.core.controllers.trainer import Evaluator
from fastNLP.core.metrics.accuracy import Accuracy
from fastNLP.core.callbacks.progress_callback import RichCallback
from fastNLP.core.dataloaders.jittor_dataloader.fdl import JittorDataLoader
from fastNLP.envs.imports import _NEED_IMPORT_JITTOR

if _NEED_IMPORT_JITTOR:
    import jittor as jt
    from jittor import nn, Module
    from jittor.dataset import Dataset
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Module
    from fastNLP.core.utils.dummy_class import DummyClass as Dataset
jt.flags.use_cuda=1


class JittorNormalModel_Classification(Module):
    """
    基础的 Jittor 分类模型
    """

    def __init__(self, num_labels, feature_dimension):
        super(JittorNormalModel_Classification, self).__init__()
        self.num_labels = num_labels

        self.linear1 = nn.Linear(in_features=feature_dimension, out_features=64)
        self.ac1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=64, out_features=32)
        self.ac2 = nn.ReLU()
        self.output = nn.Linear(in_features=32, out_features=num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def execute(self, x):
        # It's similar to forward function in Pytorch
        x = self.ac1(self.linear1(x))
        x = self.ac2(self.linear2(x))
        x = self.output(x)
        return x

    def train_step(self, x, y):
        x = self(x)
        return {"loss": self.loss_fn(x, y)}

    def evaluate_step(self, x, y):
        x = self(x)
        return {"pred": x, "target": y.reshape((-1,))}


class JittorRandomMaxDataset(Dataset):
    def __init__(self, num_samples, num_features):
        super(JittorRandomMaxDataset, self).__init__()
        self.x = jt.randn((num_samples, num_features))
        self.y = self.x.argmax(dim=1)[0]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, item):
        return {"x": self.x[item], "y": self.y[item]}


class TrainJittorConfig:
    num_labels: int = 5
    feature_dimension: int = 5
    lr = 1e-1
    batch_size: int = 4
    shuffle: bool = True

@pytest.mark.parametrize("driver", ["jittor"])
@pytest.mark.parametrize("device", ["cpu", "gpu", "cuda", None])
@pytest.mark.parametrize("callbacks", [[RichCallback(100)]])
def test_trainer_jittor(
        driver,
        device,
        callbacks,
        n_epochs=3,
):
    model = JittorNormalModel_Classification(
        num_labels=TrainJittorConfig.num_labels,
        feature_dimension=TrainJittorConfig.feature_dimension
    )
    optimizer = nn.SGD(model.parameters(), lr=TrainJittorConfig.lr)
    train_dataloader = JittorDataLoader(
        dataset=JittorRandomMaxDataset(20, TrainJittorConfig.feature_dimension),
        batch_size=TrainJittorConfig.batch_size,
        shuffle=True,
        # num_workers=4,
    )
    val_dataloader = JittorDataLoader(
        dataset=JittorRandomMaxDataset(12, TrainJittorConfig.feature_dimension),
        batch_size=TrainJittorConfig.batch_size,
        shuffle=True,
        # num_workers=4,
    )
    test_dataloader = JittorDataLoader(
        dataset=JittorRandomMaxDataset(12, TrainJittorConfig.feature_dimension),
        batch_size=TrainJittorConfig.batch_size,
        shuffle=True,
        # num_workers=4,
    )
    metrics = {"acc": Accuracy()}

    trainer = Trainer(
        model=model,
        driver=driver,
        device=device,
        optimizers=optimizer,
        train_dataloader=train_dataloader,
        evaluate_dataloaders=val_dataloader,
        validate_every=-1,
        evaluate_fn="evaluate_step",
        input_mapping=None,
        output_mapping=None,
        metrics=metrics,
        n_epochs=n_epochs,
        callbacks=callbacks,
        # progress_bar="rich"
    )
    trainer.run()

    evaluator = Evaluator(
        model=model,
        driver=driver,
        dataloaders=test_dataloader,
        evaluate_fn="evaluate_step",
        metrics=metrics,
    )
    metric_results = evaluator.run()
    # assert metric_results["acc#acc"] > 0.80


if __name__ == "__main__":
    # test_trainer_jittor("jittor", "cpu", [RichCallback(100)])         # 测试 CPU
    # test_trainer_jittor("jittor", "cuda:0", [RichCallback(100)])      # 测试 单卡 GPU
    # test_trainer_jittor("jittor", 1, [RichCallback(100)])             # 测试 指定 GPU
    # test_trainer_jittor("jittor", [0, 1], [RichCallback(100)])        # 测试 多卡 GPU
    pytest.main(['test_trainer_jittor.py'])  # 只运行此模块

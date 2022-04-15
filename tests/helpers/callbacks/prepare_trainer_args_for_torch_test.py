
"""
这个文件主要用于提供测试 callback 时的 Trainer 的参数，可以直接使用进行对Trainer进行初始化。只需要再额外传入相应的callback就可以运行

"""

from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from fastNLP.core.metrics import Accuracy


if _NEED_IMPORT_TORCH:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    class DataSet:
        def __init__(self, num_samples=1000, num_features=10):
            g = torch.Generator()
            g.manual_seed(1000)
            self.data = torch.randn(num_samples, num_features, generator=g)
            self.y = self.data.argmax(dim=-1)

        def __getitem__(self, item):
            return {'x': self.data[item], 'target': self.y[item]}

        def __len__(self):
            return len(self.data)


    class Model(nn.Module):
        def __init__(self, num_features=5):
            super().__init__()
            self.mlps = nn.Sequential(
                nn.Linear(num_features, 20),
                nn.ReLU(),
                nn.Linear(20, 20),
                nn.Dropout(p=0.3),
                nn.ReLU(),
                nn.Linear(20, num_features)
            )

        def forward(self, x, target):
            y = self.mlps(x)
            if self.training:
                return {'loss': F.cross_entropy(y, target)}
            return {'pred': y}


def get_trainer_args(num_features=5, num_samples=20, bsz=4, lr=0.1, n_epochs=5, device=None):
    ds = DataSet(num_samples=num_samples, num_features=num_features)
    dl = DataLoader(ds, batch_size=bsz)
    model = Model(num_features=num_features)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    kwargs = {
        'model': model,
        'driver': 'torch',
        'device': device,
        'optimizers': optimizer,
        'train_dataloader': dl,
        'evaluate_dataloaders': dl,
        'metrics': {'acc': Accuracy()},
        'n_epochs': n_epochs
    }

    return kwargs
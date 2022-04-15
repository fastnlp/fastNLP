import sys
import os
import warnings
warnings.filterwarnings("ignore")
os.environ["FASTNLP_BACKEND"] = "torch"
sys.path.append("../../../../")

import paddle
from fastNLP.core.samplers import RandomSampler
from fastNLP.core.drivers.paddle_driver.utils import replace_sampler, replace_batch_sampler
from tests.helpers.datasets.paddle_data import PaddleNormalDataset

dataset = PaddleNormalDataset(20)
batch_sampler = paddle.io.BatchSampler(dataset=dataset, batch_size=2)
batch_sampler.sampler = RandomSampler(dataset, True)
dataloader = paddle.io.DataLoader(
    dataset,
    batch_sampler=batch_sampler
)

forward_steps = 9
iter_dataloader = iter(dataloader)
for _ in range(forward_steps):
    print(next(iter_dataloader))
print(dataloader.batch_sampler.sampler.during_iter)

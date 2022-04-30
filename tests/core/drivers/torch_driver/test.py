import sys
sys.path.append("../../../../")
from fastNLP.core.drivers.torch_driver.ddp import TorchDDPDriver
from tests.helpers.models.torch_model import TorchNormalModel_Classification_1

import torch

device = [0, 1]
torch_model = TorchNormalModel_Classification_1(10, 10)
torch_opt = torch.optim.Adam(params=torch_model.parameters(), lr=0.01)
device = [torch.device(i) for i in device]
driver = TorchDDPDriver(
    model=torch_model,
    parallel_device=device,
    fp16=False
)
driver.set_optimizers(torch_opt)
driver.setup()
print("-----------first--------------")

device = [0, 2]
torch_model = TorchNormalModel_Classification_1(10, 10)
torch_opt = torch.optim.Adam(params=torch_model.parameters(), lr=0.01)
device = [torch.device(i) for i in device]
driver = TorchDDPDriver(
    model=torch_model,
    parallel_device=device,
    fp16=False
)
driver.set_optimizers(torch_opt)
driver.setup()
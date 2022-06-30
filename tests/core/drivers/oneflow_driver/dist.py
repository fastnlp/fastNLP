import oneflow
from oneflow import nn
from oneflow.utils.data import DataLoader, Dataset
from oneflow.nn.parallel import DistributedDataParallel as ddp
import os
# print(oneflow.ones(3,4).device)
# print(oneflow.rand(3,4).device)
# exit(0)
# PLACEMENT = oneflow.placement("cuda", [0,1])
# S0 = oneflow.sbp.split(0)
# B = oneflow.sbp.broadcast
print(oneflow.cuda.current_device())
exit(0)
class OneflowArgMaxDataset(Dataset):
    def __init__(self, feature_dimension=10, data_num=1000, seed=0):
        self.num_labels = feature_dimension
        self.feature_dimension = feature_dimension
        self.data_num = data_num
        self.seed = seed

        g = oneflow.Generator()
        g.manual_seed(1000)
        self.x = oneflow.randint(low=-100, high=100, size=[data_num, feature_dimension], generator=g).float()
        self.y = oneflow.max(self.x, dim=-1)[1]

    def __len__(self):
        return self.data_num

    def __getitem__(self, item):
        return self.x[item], self.y[item]

class Model(nn.Module):
    def __init__(self, num_labels, feature_dimension):
        super(Model, self).__init__()
        self.num_labels = num_labels

        self.linear1 = nn.Linear(in_features=feature_dimension, out_features=10)
        self.ac1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=10, out_features=10)
        self.ac2 = nn.ReLU()
        self.output = nn.Linear(in_features=10, out_features=num_labels)

    def forward(self, x):
        x = self.ac1(self.linear1(x))
        x = self.ac2(self.linear2(x))
        x = self.output(x)
        return x

dataset = OneflowArgMaxDataset(10, 100)
model = Model(10, 10)
loss_func = nn.CrossEntropyLoss()
optimizer = oneflow.optim.Adam(model.parameters(), 0.001)
dataloader = oneflow.utils.data.DataLoader(dataset, batch_size=32)

device = "cuda"
model.to(device)
# model = ddp(model)
loss_func.to(device)

# model = model.to_global(PLACEMENT, B)

for i in range(2):
    for i, (x, y) in enumerate(dataloader):
        if i % 2 != oneflow.env.get_rank():
            continue
        x = x.to(device)
        y = y.to(device)
        # x = x.to_global(PLACEMENT, S0)
        # y = y.to_global(PLACEMENT, S0)
        output = model(x)
        loss = loss_func(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
oneflow.save(model, "ttt")
print("end.")
# python -m oneflow.distributed.launch --nproc_per_node 2 dist.py


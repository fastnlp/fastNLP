# 首先需要加入以下的路径到环境变量，因为当前只对内部测试开放，所以需要手动申明一下路径

import os
import sys
sys.path.append('../../')
os.environ['FASTNLP_BASE_URL'] = 'http://10.141.222.118:8888/file/download/'
os.environ['FASTNLP_CACHE_DIR'] = '/remote-home/hyan01/fastnlp_caches'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

from fastNLP.core.const import Const as C
from fastNLP.core import LRScheduler
from fastNLP.io.data_loader import YelpLoader
from reproduction.text_classification.model.HAN import HANCLS
from fastNLP.embeddings import StaticEmbedding
from fastNLP import CrossEntropyLoss, AccuracyMetric
from fastNLP.core.trainer import Trainer
from torch.optim import SGD
import torch.cuda
from torch.optim.lr_scheduler import CosineAnnealingLR


##hyper

class Config():
    model_dir_or_name = "en-base-uncased"
    embedding_grad = False,
    train_epoch = 30
    batch_size = 100
    num_classes = 5
    task = "yelp"
    #datadir = '/remote-home/lyli/fastNLP/yelp_polarity/'
    datadir = '/remote-home/ygwang/yelp_polarity/'
    datafile = {"train": "train.csv", "test": "test.csv"}
    lr = 1e-3

    def __init__(self):
        self.datapath = {k: os.path.join(self.datadir, v)
                         for k, v in self.datafile.items()}


ops = Config()

##1.task相关信息：利用dataloader载入dataInfo

datainfo = YelpLoader(fine_grained=True).process(paths=ops.datapath, train_ds=['train'])
print(len(datainfo.datasets['train']))
print(len(datainfo.datasets['test']))


# post process
def make_sents(words):
    sents = [words]
    return sents


for dataset in datainfo.datasets.values():
    dataset.apply_field(make_sents, field_name='words', new_field_name='input_sents')

datainfo = datainfo
datainfo.datasets['train'].set_input('input_sents')
datainfo.datasets['test'].set_input('input_sents')
datainfo.datasets['train'].set_target('target')
datainfo.datasets['test'].set_target('target')

## 2.或直接复用fastNLP的模型

vocab = datainfo.vocabs['words']
# embedding = StackEmbedding([StaticEmbedding(vocab), CNNCharEmbedding(vocab, 100)])
embedding = StaticEmbedding(vocab)

print(len(vocab))
print(len(datainfo.vocabs['target']))

# model = DPCNN(init_embed=embedding, num_cls=ops.num_classes)
model = HANCLS(init_embed=embedding, num_cls=ops.num_classes)

## 3. 声明loss,metric,optimizer
loss = CrossEntropyLoss(pred=C.OUTPUT, target=C.TARGET)
metric = AccuracyMetric(pred=C.OUTPUT, target=C.TARGET)
optimizer = SGD([param for param in model.parameters() if param.requires_grad == True],
                lr=ops.lr, momentum=0.9, weight_decay=0)

callbacks = []
callbacks.append(LRScheduler(CosineAnnealingLR(optimizer, 5)))

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)

for ds in datainfo.datasets.values():
    ds.apply_field(len, C.INPUT, C.INPUT_LEN)
    ds.set_input(C.INPUT, C.INPUT_LEN)
    ds.set_target(C.TARGET)


## 4.定义train方法
def train(model, datainfo, loss, metrics, optimizer, num_epochs=ops.train_epoch):
    trainer = Trainer(datainfo.datasets['train'], model, optimizer=optimizer, loss=loss,
                      metrics=[metrics], dev_data=datainfo.datasets['test'], device=device,
                      check_code_level=-1, batch_size=ops.batch_size, callbacks=callbacks,
                      n_epochs=num_epochs)

    print(trainer.train())


if __name__ == "__main__":
    train(model, datainfo, loss, metric, optimizer)

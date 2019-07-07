# 首先需要加入以下的路径到环境变量，因为当前只对内部测试开放，所以需要手动申明一下路径
import os
os.environ['FASTNLP_BASE_URL'] = 'http://10.141.222.118:8888/file/download/'
os.environ['FASTNLP_CACHE_DIR'] = '/remote-home/hyan01/fastnlp_caches'


import torch.nn as nn

from data.IMDBLoader import IMDBLoader
from fastNLP.modules.encoder.embedding import StaticEmbedding
from model.lstm_self_attention import BiLSTM_SELF_ATTENTION

from fastNLP.core.const import Const as C
from fastNLP import CrossEntropyLoss, AccuracyMetric
from fastNLP import Trainer, Tester
from torch.optim import Adam
from fastNLP.io.model_io import ModelLoader, ModelSaver

import argparse


class Config():
    train_epoch= 10
    lr=0.001

    num_classes=2
    hidden_dim=256
    num_layers=1
    attention_unit=256
    attention_hops=1
    nfc=128

    task_name = "IMDB"
    datapath={"train":"IMDB_data/train.csv", "test":"IMDB_data/test.csv"}
    load_model_path="./result_IMDB/best_BiLSTM_SELF_ATTENTION_acc_2019-07-07-04-16-51"
    save_model_path="./result_IMDB_test/"
opt=Config


# load data
dataloader=IMDBLoader()
datainfo=dataloader.process(opt.datapath)

# print(datainfo.datasets["train"])
# print(datainfo)


# define model
vocab=datainfo.vocabs['words']
embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-840b-300', requires_grad=True)
model=BiLSTM_SELF_ATTENTION(init_embed=embed, num_classes=opt.num_classes, hidden_dim=opt.hidden_dim, num_layers=opt.num_layers, attention_unit=opt.attention_unit, attention_hops=opt.attention_hops, nfc=opt.nfc)


# define loss_function and metrics
loss=CrossEntropyLoss()
metrics=AccuracyMetric()
optimizer= Adam([param for param in model.parameters() if param.requires_grad==True], lr=opt.lr)


def train(datainfo, model, optimizer, loss, metrics, opt):
    trainer = Trainer(datainfo.datasets['train'], model, optimizer=optimizer, loss=loss,
                        metrics=metrics, dev_data=datainfo.datasets['test'], device=0, check_code_level=-1,
                        n_epochs=opt.train_epoch, save_path=opt.save_model_path)
    trainer.train()


if __name__ == "__main__":
    train(datainfo, model, optimizer, loss, metrics, opt)

# 这个模型需要在pytorch=0.4下运行，weight_drop不支持1.0

# 首先需要加入以下的路径到环境变量，因为当前只对内部测试开放，所以需要手动申明一下路径
import os
os.environ['FASTNLP_BASE_URL'] = 'http://10.141.222.118:8888/file/download/'
os.environ['FASTNLP_CACHE_DIR'] = '/remote-home/hyan01/fastnlp_caches'


import torch.nn as nn

from data.SSTLoader import SSTLoader
from data.IMDBLoader import IMDBLoader
from data.yelpLoader import yelpLoader
from fastNLP.modules.encoder.embedding import StaticEmbedding
from model.awd_lstm import AWDLSTMSentiment

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
    nfc=128
    wdrop=0.5

    task_name = "IMDB"
    datapath={"train":"IMDB_data/train.csv", "test":"IMDB_data/test.csv"}
    load_model_path="./result_IMDB/best_BiLSTM_SELF_ATTENTION_acc_2019-07-07-04-16-51"
    save_model_path="./result_IMDB_test/"
opt=Config


# load data
dataloaders = {
    "IMDB":IMDBLoader(),
    "YELP":yelpLoader(),
    "SST-5":SSTLoader(subtree=True,fine_grained=True),
    "SST-3":SSTLoader(subtree=True,fine_grained=False)
}

if opt.task_name not in ["IMDB", "YELP", "SST-5", "SST-3"]:
    raise ValueError("task name must in ['IMDB', 'YELP, 'SST-5', 'SST-3']")

dataloader = dataloaders[opt.task_name]
datainfo=dataloader.process(opt.datapath)
# print(datainfo.datasets["train"])
# print(datainfo)


# define model
vocab=datainfo.vocabs['words']
embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-840b-300', requires_grad=True)
model=AWDLSTMSentiment(init_embed=embed, num_classes=opt.num_classes, hidden_dim=opt.hidden_dim, num_layers=opt.num_layers, nfc=opt.nfc, wdrop=opt.wdrop)


# define loss_function and metrics
loss=CrossEntropyLoss()
metrics=AccuracyMetric()
optimizer= Adam([param for param in model.parameters() if param.requires_grad==True], lr=opt.lr)


def train(datainfo, model, optimizer, loss, metrics, opt):
    trainer = Trainer(datainfo.datasets['train'], model, optimizer=optimizer, loss=loss,
                        metrics=metrics, dev_data=datainfo.datasets['dev'], device=0, check_code_level=-1,
                        n_epochs=opt.train_epoch, save_path=opt.save_model_path)
    trainer.train()


def test(datainfo, metrics, opt):
    # load model
    model = ModelLoader.load_pytorch_model(opt.load_model_path)
    print("model loaded!")

    # Tester
    tester = Tester(datainfo.datasets['test'], model, metrics, batch_size=4, device=0)
    acc = tester.test()
    print("acc=",acc)



parser = argparse.ArgumentParser()
parser.add_argument('--mode', required=True, dest="mode",help='set the model\'s model')


args = parser.parse_args()
if args.mode == 'train':
    train(datainfo, model, optimizer, loss, metrics, opt)
elif args.mode == 'test':
    test(datainfo, metrics, opt)
else:
    print('no mode specified for model!')
    parser.print_help()

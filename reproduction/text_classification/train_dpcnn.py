# 首先需要加入以下的路径到环境变量，因为当前只对内部测试开放，所以需要手动申明一下路径

import torch.cuda
from fastNLP.core.utils import cache_results
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from fastNLP.core.trainer import Trainer
from fastNLP import CrossEntropyLoss, AccuracyMetric
from fastNLP.modules.encoder.embedding import StaticEmbedding, CNNCharEmbedding, StackEmbedding
from reproduction.text_classification.model.dpcnn import DPCNN
from data.yelpLoader import yelpLoader
import torch.nn as nn
from fastNLP.core import LRScheduler
from fastNLP.core.const import Const as C
from fastNLP.core.vocabulary import VocabularyOption
from utils.util_init import set_rng_seeds
import os
os.environ['FASTNLP_BASE_URL'] = 'http://10.141.222.118:8888/file/download/'
os.environ['FASTNLP_CACHE_DIR'] = '/remote-home/hyan01/fastnlp_caches'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"



# hyper

class Config():
    seed = 12345
    model_dir_or_name = "dpcnn-yelp-p"
    embedding_grad = True
    train_epoch = 30
    batch_size = 100
    num_classes = 2
    task = "yelp_p"
    #datadir = '/remote-home/yfshao/workdir/datasets/SST'
    datadir = '/remote-home/yfshao/workdir/datasets/yelp_polarity'
    #datafile = {"train": "train.txt", "dev": "dev.txt", "test": "test.txt"}
    datafile = {"train": "train.csv",  "test": "test.csv"}
    lr = 1e-3
    src_vocab_op = VocabularyOption()
    embed_dropout = 0.3
    cls_dropout = 0.1
    weight_decay = 1e-4

    def __init__(self):
        self.datapath = {k: os.path.join(self.datadir, v)
                         for k, v in self.datafile.items()}


ops = Config()

set_rng_seeds(ops.seed)
print('RNG SEED: {}'.format(ops.seed))

# 1.task相关信息：利用dataloader载入dataInfo

#datainfo=SSTLoader(fine_grained=True).process(paths=ops.datapath, train_ds=['train'])
@cache_results(ops.model_dir_or_name+'-data-cache')
def load_data():
    datainfo = yelpLoader(fine_grained=True, lower=True).process(
        paths=ops.datapath, train_ds=['train'], src_vocab_op=ops.src_vocab_op)
    for ds in datainfo.datasets.values():
        ds.apply_field(len, C.INPUT, C.INPUT_LEN)
        ds.set_input(C.INPUT, C.INPUT_LEN)
        ds.set_target(C.TARGET)
    return datainfo

datainfo = load_data()

# 2.或直接复用fastNLP的模型

vocab = datainfo.vocabs['words']
# embedding = StackEmbedding([StaticEmbedding(vocab), CNNCharEmbedding(vocab, 100)])
#embedding = StaticEmbedding(vocab)

embedding = StaticEmbedding(
    vocab, model_dir_or_name='en-word2vec-300', requires_grad=ops.embedding_grad,
    normalize=False
)

print(len(datainfo.datasets['train']))
print(len(datainfo.datasets['test']))
print(datainfo.datasets['train'][0])


print(len(vocab))
print(len(datainfo.vocabs['target']))


model = DPCNN(init_embed=embedding, num_cls=ops.num_classes,
              embed_dropout=ops.embed_dropout, cls_dropout=ops.cls_dropout)
print(model)

# 3. 声明loss,metric,optimizer
loss = CrossEntropyLoss(pred=C.OUTPUT, target=C.TARGET)
metric = AccuracyMetric(pred=C.OUTPUT, target=C.TARGET)
optimizer = SGD([param for param in model.parameters() if param.requires_grad == True],
                lr=ops.lr, momentum=0.9, weight_decay=ops.weight_decay)

callbacks = []
callbacks.append(LRScheduler(CosineAnnealingLR(optimizer, 5)))
# callbacks.append
#     LRScheduler(LambdaLR(optimizer, lambda epoch: ops.lr if epoch <
#                          ops.train_epoch * 0.8 else ops.lr * 0.1))
# )

# callbacks.append(
#     FitlogCallback(data=datainfo.datasets, verbose=1)
# )

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)

# 4.定义train方法
trainer = Trainer(datainfo.datasets['train'], model, optimizer=optimizer, loss=loss,
                  metrics=[metric],
                  dev_data=datainfo.datasets['test'], device=device,
                  check_code_level=-1, batch_size=ops.batch_size, callbacks=callbacks,
                  n_epochs=ops.train_epoch, num_workers=4)



if __name__ == "__main__":
    print(trainer.train())


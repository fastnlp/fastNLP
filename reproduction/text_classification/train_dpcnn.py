# 首先需要加入以下的路径到环境变量，因为当前只对内部测试开放，所以需要手动申明一下路径

import torch.cuda
from fastNLP.core.utils import cache_results
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from fastNLP.core.trainer import Trainer
from fastNLP import CrossEntropyLoss, AccuracyMetric
from fastNLP.embeddings import StaticEmbedding
from reproduction.text_classification.model.dpcnn import DPCNN
from fastNLP.core.sampler import BucketSampler
from fastNLP.core import LRScheduler
from fastNLP.core.const import Const as C
from fastNLP.core.vocabulary import VocabularyOption
from utils.util_init import set_rng_seeds
from fastNLP import logger
import os
from fastNLP.io import YelpFullPipe, YelpPolarityPipe

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# hyper
logger.add_file('log', 'INFO')

class Config():
    seed = 12345
    model_dir_or_name = "dpcnn-yelp-f"
    embedding_grad = True
    train_epoch = 30
    batch_size = 100
    task = "yelp_f"
    #datadir = 'workdir/datasets/SST'
    # datadir = 'workdir/datasets/yelp_polarity'
    datadir = 'workdir/datasets/yelp_full'
    #datafile = {"train": "train.txt", "dev": "dev.txt", "test": "test.txt"}
    datafile = {"train": "train.csv",  "test": "test.csv"}
    lr = 1e-3
    src_vocab_op = VocabularyOption(max_size=100000)
    embed_dropout = 0.3
    cls_dropout = 0.1
    weight_decay = 1e-5

    def __init__(self):
        self.datadir = os.path.join(os.environ['HOME'], self.datadir)
        self.datapath = {k: os.path.join(self.datadir, v)
                         for k, v in self.datafile.items()}

ops = Config()

set_rng_seeds(ops.seed)
logger.info('RNG SEED %d'%ops.seed)

# 1.task相关信息：利用dataloader载入dataInfo


@cache_results(ops.model_dir_or_name+'-data-cache')
def load_data():
    datainfo = YelpFullPipe(lower=True, tokenizer='raw').process_from_file(ops.datapath)
    for ds in datainfo.datasets.values():
        ds.apply_field(len, C.INPUT, C.INPUT_LEN)
        ds.set_input(C.INPUT, C.INPUT_LEN)
        ds.set_target(C.TARGET)

    return datainfo


datainfo = load_data()
embedding = StaticEmbedding(
        datainfo.vocabs['words'], model_dir_or_name='en-glove-6b-100d', requires_grad=ops.embedding_grad,
        normalize=False)
embedding.embedding.weight.data /= embedding.embedding.weight.data.std()
print(embedding.embedding.weight.data.mean(), embedding.embedding.weight.data.std())

# 2.或直接复用fastNLP的模型

# datainfo.datasets['train'] = datainfo.datasets['train'][:1000] # for debug purpose
# datainfo.datasets['test'] = datainfo.datasets['test'][:1000]
logger.info(datainfo)

model = DPCNN(init_embed=embedding, num_cls=len(datainfo.vocabs[C.TARGET]),
              embed_dropout=ops.embed_dropout, cls_dropout=ops.cls_dropout)
# print(model)

# 3. 声明loss,metric,optimizer
loss = CrossEntropyLoss(pred=C.OUTPUT, target=C.TARGET)
metric = AccuracyMetric(pred=C.OUTPUT, target=C.TARGET)
optimizer = SGD([param for param in model.parameters() if param.requires_grad == True],
                lr=ops.lr, momentum=0.9, weight_decay=ops.weight_decay)

callbacks = []

callbacks.append(LRScheduler(CosineAnnealingLR(optimizer, 5)))


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# print(device)
logger.info(device)

# 4.定义train方法
# normal trainer
trainer = Trainer(datainfo.datasets['train'], model, optimizer=optimizer, loss=loss,
                  sampler=BucketSampler(num_buckets=50, batch_size=ops.batch_size),
                  metrics=[metric], use_tqdm=False, save_path='save',
                  dev_data=datainfo.datasets['test'], device=device,
                  check_code_level=-1, batch_size=ops.batch_size, callbacks=callbacks,
                  n_epochs=ops.train_epoch, num_workers=4)

# distributed trainer
# trainer = DistTrainer(datainfo.datasets['train'], model, optimizer=optimizer, loss=loss,
#                       metrics=[metric],
#                       dev_data=datainfo.datasets['test'], device='cuda',
#                       batch_size_per_gpu=ops.batch_size, callbacks_all=callbacks,
#                       n_epochs=ops.train_epoch, num_workers=4)


if __name__ == "__main__":
    print(trainer.train())

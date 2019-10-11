
import random
import numpy as np
import torch
from torch.optim import Adamax
from torch.optim.lr_scheduler import StepLR

from fastNLP.core import Trainer, Tester, AccuracyMetric, Const
from fastNLP.core.callback import GradientClipCallback, LRScheduler, EvaluateCallback
from fastNLP.core.losses import CrossEntropyLoss
from fastNLP.embeddings import StaticEmbedding
from fastNLP.embeddings import ElmoEmbedding
from fastNLP.io.pipe.matching import SNLIPipe, RTEPipe, MNLIPipe, QNLIPipe, QuoraPipe
from fastNLP.models.snli import ESIM


# define hyper-parameters
class ESIMConfig:

    task = 'snli'

    embedding = 'glove'

    batch_size_per_gpu = 196
    n_epochs = 30
    lr = 2e-3
    seed = 42
    save_path = None  # 模型存储的位置，None表示不存储模型。

    train_dataset_name = 'train'
    dev_dataset_name = 'dev'
    test_dataset_name = 'test'

    to_lower = True  # 忽略大小写
    tokenizer = 'spacy'  # 使用spacy进行分词


arg = ESIMConfig()

# set random seed
random.seed(arg.seed)
np.random.seed(arg.seed)
torch.manual_seed(arg.seed)

n_gpu = torch.cuda.device_count()
if n_gpu > 0:
    torch.cuda.manual_seed_all(arg.seed)

# load data set
if arg.task == 'snli':
    data_bundle = SNLIPipe(lower=arg.to_lower, tokenizer=arg.tokenizer).process_from_file()
elif arg.task == 'rte':
    data_bundle = RTEPipe(lower=arg.to_lower, tokenizer=arg.tokenizer).process_from_file()
elif arg.task == 'qnli':
    data_bundle = QNLIPipe(lower=arg.to_lower, tokenizer=arg.tokenizer).process_from_file()
elif arg.task == 'mnli':
    data_bundle = MNLIPipe(lower=arg.to_lower, tokenizer=arg.tokenizer).process_from_file()
elif arg.task == 'quora':
    data_bundle = QuoraPipe(lower=arg.to_lower, tokenizer=arg.tokenizer).process_from_file()
else:
    raise RuntimeError(f'NOT support {arg.task} task yet!')

print(data_bundle)  # print details in data_bundle

# load embedding
if arg.embedding == 'elmo':
    embedding = ElmoEmbedding(data_bundle.vocabs[Const.INPUTS(0)], model_dir_or_name='en-medium',
                              requires_grad=True)
elif arg.embedding == 'glove':
    embedding = StaticEmbedding(data_bundle.vocabs[Const.INPUTS(0)], model_dir_or_name='en-glove-840b-300d',
                                requires_grad=True, normalize=False)
else:
    raise RuntimeError(f'NOT support {arg.embedding} embedding yet!')

# define model
model = ESIM(embedding, num_labels=len(data_bundle.vocabs[Const.TARGET]))

# define optimizer and callback
optimizer = Adamax(lr=arg.lr, params=model.parameters())
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # 每10个epoch学习率变为原来的0.5倍

callbacks = [
    GradientClipCallback(clip_value=10),  # 等价于torch.nn.utils.clip_grad_norm_(10)
    LRScheduler(scheduler),
]

if arg.task in ['snli']:
    callbacks.append(EvaluateCallback(data=data_bundle.datasets[arg.test_dataset_name]))
    # evaluate test set in every epoch if task is snli.

# define trainer
trainer = Trainer(train_data=data_bundle.datasets[arg.train_dataset_name], model=model,
                  optimizer=optimizer,
                  loss=CrossEntropyLoss(),
                  batch_size=torch.cuda.device_count() * arg.batch_size_per_gpu,
                  n_epochs=arg.n_epochs, print_every=-1,
                  dev_data=data_bundle.datasets[arg.dev_dataset_name],
                  metrics=AccuracyMetric(), metric_key='acc',
                  device=[i for i in range(torch.cuda.device_count())],
                  check_code_level=-1,
                  save_path=arg.save_path,
                  callbacks=callbacks)

# train model
trainer.train(load_best_model=True)

# define tester
tester = Tester(
    data=data_bundle.datasets[arg.test_dataset_name],
    model=model,
    metrics=AccuracyMetric(),
    batch_size=torch.cuda.device_count() * arg.batch_size_per_gpu,
    device=[i for i in range(torch.cuda.device_count())],
)

# test model
tester.test()



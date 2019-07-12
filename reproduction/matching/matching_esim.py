
import random
import numpy as np
import torch
from torch.optim import Adamax
from torch.optim.lr_scheduler import StepLR

from fastNLP.core import Trainer, Tester, AccuracyMetric, Const
from fastNLP.core.callback import GradientClipCallback, LRScheduler
from fastNLP.embeddings.static_embedding import StaticEmbedding
from fastNLP.embeddings.elmo_embedding import ElmoEmbedding
from fastNLP.io.data_loader import SNLILoader, RTELoader, MNLILoader, QNLILoader, QuoraLoader
from fastNLP.models.snli import ESIM


# define hyper-parameters
class ESIMConfig:

    task = 'snli'
    embedding = 'glove'
    batch_size_per_gpu = 196
    n_epochs = 30
    lr = 2e-3
    seq_len_type = 'seq_len'
    # seq_len表示在process的时候用len(words)来表示长度信息；
    # mask表示用0/1掩码矩阵来表示长度信息；
    seed = 42
    train_dataset_name = 'train'
    dev_dataset_name = 'dev'
    test_dataset_name = 'test'
    save_path = None  # 模型存储的位置，None表示不存储模型。


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
    data_info = SNLILoader().process(
        paths='path/to/snli/data', to_lower=False, seq_len_type=arg.seq_len_type,
        get_index=True, concat=False,
    )
elif arg.task == 'rte':
    data_info = RTELoader().process(
        paths='path/to/rte/data', to_lower=False, seq_len_type=arg.seq_len_type,
        get_index=True, concat=False,
    )
elif arg.task == 'qnli':
    data_info = QNLILoader().process(
        paths='path/to/qnli/data', to_lower=False, seq_len_type=arg.seq_len_type,
        get_index=True, concat=False,
    )
elif arg.task == 'mnli':
    data_info = MNLILoader().process(
        paths='path/to/mnli/data', to_lower=False, seq_len_type=arg.seq_len_type,
        get_index=True, concat=False,
    )
elif arg.task == 'quora':
    data_info = QuoraLoader().process(
        paths='path/to/quora/data', to_lower=False, seq_len_type=arg.seq_len_type,
        get_index=True, concat=False,
    )
else:
    raise RuntimeError(f'NOT support {arg.task} task yet!')

# load embedding
if arg.embedding == 'elmo':
    embedding = ElmoEmbedding(data_info.vocabs[Const.INPUT], requires_grad=True)
elif arg.embedding == 'glove':
    embedding = StaticEmbedding(data_info.vocabs[Const.INPUT], requires_grad=True, normalize=False)
else:
    raise RuntimeError(f'NOT support {arg.embedding} embedding yet!')

# define model
model = ESIM(embedding, num_labels=len(data_info.vocabs[Const.TARGET]))

# define optimizer and callback
optimizer = Adamax(lr=arg.lr, params=model.parameters())
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # 每10个epoch学习率变为原来的0.5倍

callbacks = [
    GradientClipCallback(clip_value=10),  # 等价于torch.nn.utils.clip_grad_norm_(10)
    LRScheduler(scheduler),
]

# define trainer
trainer = Trainer(train_data=data_info.datasets[arg.train_dataset_name], model=model,
                  optimizer=optimizer,
                  batch_size=torch.cuda.device_count() * arg.batch_size_per_gpu,
                  n_epochs=arg.n_epochs, print_every=-1,
                  dev_data=data_info.datasets[arg.dev_dataset_name],
                  metrics=AccuracyMetric(), metric_key='acc',
                  device=[i for i in range(torch.cuda.device_count())],
                  check_code_level=-1,
                  save_path=arg.save_path)

# train model
trainer.train(load_best_model=True)

# define tester
tester = Tester(
    data=data_info.datasets[arg.test_dataset_name],
    model=model,
    metrics=AccuracyMetric(),
    batch_size=torch.cuda.device_count() * arg.batch_size_per_gpu,
    device=[i for i in range(torch.cuda.device_count())],
)

# test model
tester.test()



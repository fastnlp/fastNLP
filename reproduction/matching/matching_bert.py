import random
import numpy as np
import torch

from fastNLP.core import Trainer, Tester, AccuracyMetric, Const
from fastNLP.core.callback import WarmupCallback, EvaluateCallback
from fastNLP.core.optimizer import AdamW
from fastNLP.embeddings import BertEmbedding
from fastNLP.io.pipe.matching import SNLIBertPipe, RTEBertPipe, MNLIBertPipe,\
    QNLIBertPipe, QuoraBertPipe
from fastNLP.models.bert import BertForSentenceMatching


# define hyper-parameters
class BERTConfig:

    task = 'snli'

    batch_size_per_gpu = 6
    n_epochs = 6
    lr = 2e-5
    warm_up_rate = 0.1
    seed = 42
    save_path = None  # 模型存储的位置，None表示不存储模型。

    train_dataset_name = 'train'
    dev_dataset_name = 'dev'
    test_dataset_name = 'test'

    to_lower = True  # 忽略大小写
    tokenizer = 'spacy'  # 使用spacy进行分词

    bert_model_dir_or_name = 'bert-base-uncased'


arg = BERTConfig()

# set random seed
random.seed(arg.seed)
np.random.seed(arg.seed)
torch.manual_seed(arg.seed)

n_gpu = torch.cuda.device_count()
if n_gpu > 0:
    torch.cuda.manual_seed_all(arg.seed)

# load data set
if arg.task == 'snli':
    data_bundle = SNLIBertPipe(lower=arg.to_lower, tokenizer=arg.tokenizer).process_from_file()
elif arg.task == 'rte':
    data_bundle = RTEBertPipe(lower=arg.to_lower, tokenizer=arg.tokenizer).process_from_file()
elif arg.task == 'qnli':
    data_bundle = QNLIBertPipe(lower=arg.to_lower, tokenizer=arg.tokenizer).process_from_file()
elif arg.task == 'mnli':
    data_bundle = MNLIBertPipe(lower=arg.to_lower, tokenizer=arg.tokenizer).process_from_file()
elif arg.task == 'quora':
    data_bundle = QuoraBertPipe(lower=arg.to_lower, tokenizer=arg.tokenizer).process_from_file()
else:
    raise RuntimeError(f'NOT support {arg.task} task yet!')

print(data_bundle)  # print details in data_bundle

# load embedding
embed = BertEmbedding(data_bundle.vocabs[Const.INPUT], model_dir_or_name=arg.bert_model_dir_or_name)

# define model
model = BertForSentenceMatching(embed, num_labels=len(data_bundle.vocabs[Const.TARGET]))

# define optimizer and callback
optimizer = AdamW(lr=arg.lr, params=model.parameters())
callbacks = [WarmupCallback(warmup=arg.warm_up_rate, schedule='linear'), ]

if arg.task in ['snli']:
    callbacks.append(EvaluateCallback(data=data_bundle.datasets[arg.test_dataset_name]))
    # evaluate test set in every epoch if task is snli.

# define trainer
trainer = Trainer(train_data=data_bundle.get_dataset(arg.train_dataset_name), model=model,
                  optimizer=optimizer,
                  batch_size=torch.cuda.device_count() * arg.batch_size_per_gpu,
                  n_epochs=arg.n_epochs, print_every=-1,
                  dev_data=data_bundle.get_dataset(arg.dev_dataset_name),
                  metrics=AccuracyMetric(), metric_key='acc',
                  device=[i for i in range(torch.cuda.device_count())],
                  check_code_level=-1,
                  save_path=arg.save_path,
                  callbacks=callbacks)

# train model
trainer.train(load_best_model=True)

# define tester
tester = Tester(
    data=data_bundle.get_dataset(arg.test_dataset_name),
    model=model,
    metrics=AccuracyMetric(),
    batch_size=torch.cuda.device_count() * arg.batch_size_per_gpu,
    device=[i for i in range(torch.cuda.device_count())],
)

# test model
tester.test()



import os

import torch

from fastNLP.core import Trainer, Tester, Adam, AccuracyMetric, Const

from fastNLP.io.dataset_loader import MatchingLoader

from reproduction.matching.model.bert import BertForNLI
from reproduction.matching.model.esim import ESIMModel


bert_dirs = 'path/to/bert/dir'

# load data set
# data_info = MatchingLoader(data_format='snli', for_model='bert', bert_dir=bert_dirs).process(...
data_info = MatchingLoader(data_format='snli', for_model='esim').process(
    {'train': './data/snli/snli_1.0_train.jsonl',
     'dev': './data/snli/snli_1.0_dev.jsonl',
     'test': './data/snli/snli_1.0_test.jsonl'},
    input_field=[Const.TARGET]
)

# model = BertForNLI(bert_dir=bert_dirs)
model = ESIMModel(data_info.embeddings['elmo'],)

trainer = Trainer(train_data=data_info.datasets['train'], model=model,
                  optimizer=Adam(lr=1e-4, model_params=model.parameters()),
                  batch_size=torch.cuda.device_count() * 24, n_epochs=20, print_every=-1,
                  dev_data=data_info.datasets['dev'],
                  metrics=AccuracyMetric(), metric_key='acc', device=[i for i in range(torch.cuda.device_count())],
                  check_code_level=-1)
trainer.train(load_best_model=True)

tester = Tester(
    data=data_info.datasets['test'],
    model=model,
    metrics=AccuracyMetric(),
    batch_size=torch.cuda.device_count() * 12,
    device=[i for i in range(torch.cuda.device_count())],
)
tester.test()



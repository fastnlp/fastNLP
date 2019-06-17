import os

import torch

from fastNLP.core import Trainer, Tester, Adam, AccuracyMetric

from fastNLP.io.dataset_loader import MatchingLoader

from reproduction.matching.model.bert import BertForNLI


# bert_dirs = 'path/to/bert/dir'
bert_dirs = '/remote-home/ygxu/BERT/BERT_English_uncased_L-12_H-768_A_12'

# load data set
data_info = MatchingLoader(data_format='snli', for_model='bert', bert_dir=bert_dirs).process(
    {#'train': './data/snli/snli_1.0_train.jsonl',
     'dev': './data/snli/snli_1.0_dev.jsonl',
     'test': './data/snli/snli_1.0_test.jsonl'}
)

print('successfully load data sets!')


model = BertForNLI(bert_dir=bert_dirs)

trainer = Trainer(train_data=data_info.datasets['dev'], model=model,
                  optimizer=Adam(lr=2e-5, model_params=model.parameters()),
                  batch_size=torch.cuda.device_count() * 12, n_epochs=4, print_every=-1,
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



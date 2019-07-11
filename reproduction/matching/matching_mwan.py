import random

import numpy as np
import torch
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR

from fastNLP import CrossEntropyLoss
from fastNLP import cache_results
from fastNLP.core import Trainer, Tester, AccuracyMetric, Const
from fastNLP.core.callback import LRScheduler, FitlogCallback
from fastNLP.embeddings import StaticEmbedding

from fastNLP.io.data_loader import MNLILoader, QNLILoader, SNLILoader, RTELoader
from reproduction.matching.model.mwan import MwanModel

import fitlog
fitlog.debug()

import argparse


argument = argparse.ArgumentParser()
argument.add_argument('--task'         , choices = ['snli', 'rte', 'qnli', 'mnli'],default = 'snli')
argument.add_argument('--batch-size'   , type = int    ,   default = 128)
argument.add_argument('--n-epochs'     , type = int    ,   default = 50)
argument.add_argument('--lr'           , type = float  ,   default = 1)
argument.add_argument('--testset-name' , type = str    ,   default = 'test')
argument.add_argument('--devset-name'  , type = str    ,   default = 'dev')
argument.add_argument('--seed'         , type = int    ,   default = 42)
argument.add_argument('--hidden-size'  , type = int    ,   default = 150)
argument.add_argument('--dropout'      , type = float  ,   default = 0.3)
arg = argument.parse_args()

random.seed(arg.seed)
np.random.seed(arg.seed)
torch.manual_seed(arg.seed)

n_gpu = torch.cuda.device_count()
if n_gpu > 0:
    torch.cuda.manual_seed_all(arg.seed)
print (n_gpu)

for k in arg.__dict__:
    print(k, arg.__dict__[k], type(arg.__dict__[k]))

# load data set
if arg.task == 'snli':
    @cache_results(f'snli_mwan.pkl')
    def read_snli():
        data_info = SNLILoader().process(
            paths='path/to/snli/data', to_lower=True, seq_len_type=None, bert_tokenizer=None,
            get_index=True, concat=False, extra_split=['/','%','-'],
        )
        return data_info
    data_info = read_snli()
elif arg.task == 'rte':
    @cache_results(f'rte_mwan.pkl')
    def read_rte():
        data_info = RTELoader().process(
            paths='path/to/rte/data', to_lower=True, seq_len_type=None, bert_tokenizer=None,
            get_index=True, concat=False, extra_split=['/','%','-'],
        )
        return data_info
    data_info = read_rte()
elif arg.task == 'qnli':
    data_info = QNLILoader().process(
        paths='path/to/qnli/data', to_lower=True, seq_len_type=None, bert_tokenizer=None,
        get_index=True, concat=False , cut_text=512, extra_split=['/','%','-'],
    )
elif arg.task == 'mnli':
    @cache_results(f'mnli_v0.9_mwan.pkl')
    def read_mnli():
        data_info = MNLILoader().process(
            paths='path/to/mnli/data', to_lower=True, seq_len_type=None, bert_tokenizer=None,
            get_index=True, concat=False, extra_split=['/','%','-'],
        )
        return data_info
    data_info = read_mnli()
else:
    raise RuntimeError(f'NOT support {arg.task} task yet!')

print(data_info)
print(len(data_info.vocabs['words']))


model = MwanModel(
	num_class = len(data_info.vocabs[Const.TARGET]), 
	EmbLayer = StaticEmbedding(data_info.vocabs[Const.INPUT], requires_grad=False, normalize=False), 
	ElmoLayer = None,
	args_of_imm = {
		"input_size"	 	: 300 , 
		"hidden_size" 		: arg.hidden_size , 
		"dropout" 			: arg.dropout	,
		"use_allennlp" 		: False , 
	} , 
)


optimizer = Adadelta(lr=arg.lr, params=model.parameters())
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

callbacks = [
    LRScheduler(scheduler),
]

if arg.task in ['snli']:
    callbacks.append(FitlogCallback(data_info.datasets[arg.testset_name], verbose=1))
elif arg.task == 'mnli':
    callbacks.append(FitlogCallback({'dev_matched': data_info.datasets['dev_matched'],
                                     'dev_mismatched': data_info.datasets['dev_mismatched']},
                                    verbose=1))

trainer = Trainer(
    train_data       = data_info.datasets['train'], 
    model            = model,
    optimizer        = optimizer, 
    num_workers      = 0,
    batch_size       = arg.batch_size,
    n_epochs         = arg.n_epochs, 
    print_every      = -1,
    dev_data         = data_info.datasets[arg.devset_name],
    metrics          = AccuracyMetric(pred = "pred" , target = "target"), 
    metric_key       = 'acc', 
    device           = [i for i in range(torch.cuda.device_count())],
    check_code_level = -1, 
    callbacks        = callbacks,
    loss             = CrossEntropyLoss(pred = "pred" , target = "target")
)
trainer.train(load_best_model=True)

tester = Tester(
    data=data_info.datasets[arg.testset_name],
    model=model,
    metrics=AccuracyMetric(),
    batch_size=arg.batch_size,
    device=[i for i in range(torch.cuda.device_count())],
)
tester.test()

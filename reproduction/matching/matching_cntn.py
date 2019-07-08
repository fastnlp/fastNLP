import argparse
import torch
import os

from fastNLP.core import Trainer, Tester, Adam, AccuracyMetric, Const
from fastNLP.modules.encoder.embedding import StaticEmbedding

from reproduction.matching.data.MatchingDataLoader import QNLILoader, RTELoader, SNLILoader, MNLILoader
from reproduction.matching.model.cntn import CNTNModel

# define hyper-parameters
argument = argparse.ArgumentParser()
argument.add_argument('--embedding', choices=['glove', 'word2vec'], default='glove')
argument.add_argument('--batch-size-per-gpu', type=int, default=256)
argument.add_argument('--n-epochs', type=int, default=200)
argument.add_argument('--lr', type=float, default=1e-5)
argument.add_argument('--seq-len-type', choices=['mask', 'seq_len'], default='mask')
argument.add_argument('--save-dir', type=str, default=None)
argument.add_argument('--cntn-depth', type=int, default=1)
argument.add_argument('--cntn-ns', type=int, default=200)
argument.add_argument('--cntn-k-top', type=int, default=10)
argument.add_argument('--cntn-r', type=int, default=5)
argument.add_argument('--dataset', choices=['qnli', 'rte', 'snli', 'mnli'], default='qnli')
argument.add_argument('--max-len', type=int, default=50)
arg = argument.parse_args()

# dataset dict
dev_dict = {
    'qnli': 'dev',
    'rte': 'dev',
    'snli': 'dev',
    'mnli': 'dev_matched',
}

test_dict = {
    'qnli': 'dev',
    'rte': 'dev',
    'snli': 'test',
    'mnli': 'dev_matched',
}

# set num_labels
if arg.dataset == 'qnli' or arg.dataset == 'rte':
    num_labels = 2
else:
    num_labels = 3

# load data set
if arg.dataset == 'qnli':
    data_info = QNLILoader().process(
        paths='path/to/qnli/data', to_lower=True, seq_len_type=arg.seq_len_type, bert_tokenizer=None,
        get_index=True, concat=False, auto_pad_length=arg.max_len)
elif arg.dataset == 'rte':
    data_info = RTELoader().process(
        paths='path/to/rte/data', to_lower=True, seq_len_type=arg.seq_len_type, bert_tokenizer=None,
        get_index=True, concat=False, auto_pad_length=arg.max_len)
elif arg.dataset == 'snli':
    data_info = SNLILoader().process(
        paths='path/to/snli/data', to_lower=True, seq_len_type=arg.seq_len_type, bert_tokenizer=None,
        get_index=True, concat=False, auto_pad_length=arg.max_len)
elif arg.dataset == 'mnli':
    data_info = MNLILoader().process(
        paths='path/to/mnli/data', to_lower=True, seq_len_type=arg.seq_len_type, bert_tokenizer=None,
        get_index=True, concat=False, auto_pad_length=arg.max_len)
else:
    raise ValueError(f'now we only support [qnli,rte,snli,mnli] dataset for cntn model!')

# load embedding
if arg.embedding == 'word2vec':
    embedding = StaticEmbedding(data_info.vocabs[Const.INPUT], model_dir_or_name='en-word2vec-300', requires_grad=True)
elif arg.embedding == 'glove':
    embedding = StaticEmbedding(data_info.vocabs[Const.INPUT], model_dir_or_name='en-glove-840b-300',
                                requires_grad=True)
else:
    raise ValueError(f'now we only support word2vec or glove embedding for cntn model!')

# define model
model = CNTNModel(embedding, ns=arg.cntn_ns, k_top=arg.cntn_k_top, num_labels=num_labels, depth=arg.cntn_depth,
                  r=arg.cntn_r)
print(model)

# define trainer
trainer = Trainer(train_data=data_info.datasets['train'], model=model,
                  optimizer=Adam(lr=arg.lr, model_params=model.parameters()),
                  batch_size=torch.cuda.device_count() * arg.batch_size_per_gpu,
                  n_epochs=arg.n_epochs, print_every=-1,
                  dev_data=data_info.datasets[dev_dict[arg.dataset]],
                  metrics=AccuracyMetric(), metric_key='acc',
                  device=[i for i in range(torch.cuda.device_count())],
                  check_code_level=-1)

# train model
trainer.train(load_best_model=True)

# define tester
tester = Tester(
    data=data_info.datasets[test_dict[arg.dataset]],
    model=model,
    metrics=AccuracyMetric(),
    batch_size=torch.cuda.device_count() * arg.batch_size_per_gpu,
    device=[i for i in range(torch.cuda.device_count())]
)

# test model
tester.test()

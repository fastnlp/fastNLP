
import argparse
import torch

from fastNLP.core import Trainer, Tester, Adam, AccuracyMetric, Const
from fastNLP.modules.encoder.embedding import ElmoEmbedding, StaticEmbedding

from reproduction.matching.data.MatchingDataLoader import SNLILoader
from reproduction.matching.model.esim import ESIMModel

argument = argparse.ArgumentParser()
argument.add_argument('--embedding', choices=['glove', 'elmo'], default='glove')
argument.add_argument('--batch-size-per-gpu', type=int, default=128)
argument.add_argument('--n-epochs', type=int, default=100)
argument.add_argument('--lr', type=float, default=1e-4)
argument.add_argument('--seq-len-type', choices=['mask', 'seq_len'], default='seq_len')
argument.add_argument('--save-dir', type=str, default=None)
arg = argument.parse_args()

bert_dirs = 'path/to/bert/dir'

# load data set
data_info = SNLILoader().process(
    paths='path/to/snli/data/dir', to_lower=True, seq_len_type=arg.seq_len_type, bert_tokenizer=None,
    get_index=True, concat=False,
)

# load embedding
if arg.embedding == 'elmo':
    embedding = ElmoEmbedding(data_info.vocabs[Const.INPUT], requires_grad=True)
elif arg.embedding == 'glove':
    embedding = StaticEmbedding(data_info.vocabs[Const.INPUT], requires_grad=True)
else:
    raise ValueError(f'now we only support elmo or glove embedding for esim model!')

# define model
model = ESIMModel(embedding)

# define trainer
trainer = Trainer(train_data=data_info.datasets['train'], model=model,
                  optimizer=Adam(lr=arg.lr, model_params=model.parameters()),
                  batch_size=torch.cuda.device_count() * arg.batch_size_per_gpu,
                  n_epochs=arg.n_epochs, print_every=-1,
                  dev_data=data_info.datasets['dev'],
                  metrics=AccuracyMetric(), metric_key='acc',
                  device=[i for i in range(torch.cuda.device_count())],
                  check_code_level=-1,
                  save_path=arg.save_path)

# train model
trainer.train(load_best_model=True)

# define tester
tester = Tester(
    data=data_info.datasets['test'],
    model=model,
    metrics=AccuracyMetric(),
    batch_size=torch.cuda.device_count() * arg.batch_size_per_gpu,
    device=[i for i in range(torch.cuda.device_count())],
)

# test model
tester.test()



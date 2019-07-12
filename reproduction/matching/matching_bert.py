import random
import numpy as np
import torch

from fastNLP.core import Trainer, Tester, AccuracyMetric, Const, Adam
from fastNLP.io.data_loader import SNLILoader, RTELoader, MNLILoader, QNLILoader, QuoraLoader

from reproduction.matching.model.bert import BertForNLI


# define hyper-parameters
class BERTConfig:

    task = 'snli'
    batch_size_per_gpu = 6
    n_epochs = 6
    lr = 2e-5
    seq_len_type = 'bert'
    seed = 42
    train_dataset_name = 'train'
    dev_dataset_name = 'dev'
    test_dataset_name = 'test'
    save_path = None  # 模型存储的位置，None表示不存储模型。
    bert_dir = 'path/to/bert/dir'  # 预训练BERT参数文件的文件夹


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
    data_info = SNLILoader().process(
        paths='path/to/snli/data', to_lower=True, seq_len_type=arg.seq_len_type,
        bert_tokenizer=arg.bert_dir, cut_text=512,
        get_index=True, concat='bert',
    )
elif arg.task == 'rte':
    data_info = RTELoader().process(
        paths='path/to/rte/data', to_lower=True, seq_len_type=arg.seq_len_type,
        bert_tokenizer=arg.bert_dir, cut_text=512,
        get_index=True, concat='bert',
    )
elif arg.task == 'qnli':
    data_info = QNLILoader().process(
        paths='path/to/qnli/data', to_lower=True, seq_len_type=arg.seq_len_type,
        bert_tokenizer=arg.bert_dir, cut_text=512,
        get_index=True, concat='bert',
    )
elif arg.task == 'mnli':
    data_info = MNLILoader().process(
        paths='path/to/mnli/data', to_lower=True, seq_len_type=arg.seq_len_type,
        bert_tokenizer=arg.bert_dir, cut_text=512,
        get_index=True, concat='bert',
    )
elif arg.task == 'quora':
    data_info = QuoraLoader().process(
        paths='path/to/quora/data', to_lower=True, seq_len_type=arg.seq_len_type,
        bert_tokenizer=arg.bert_dir, cut_text=512,
        get_index=True, concat='bert',
    )
else:
    raise RuntimeError(f'NOT support {arg.task} task yet!')

# define model
model = BertForNLI(class_num=len(data_info.vocabs[Const.TARGET]), bert_dir=arg.bert_dir)

# define trainer
trainer = Trainer(train_data=data_info.datasets[arg.train_dataset_name], model=model,
                  optimizer=Adam(lr=arg.lr, model_params=model.parameters()),
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



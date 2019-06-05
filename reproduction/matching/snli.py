import os

import torch

from fastNLP.core import Vocabulary, DataSet, Trainer, Tester, Const, Adam, AccuracyMetric

from reproduction.matching.data.SNLIDataLoader import SNLILoader
from legacy.component.bert_tokenizer import BertTokenizer
from reproduction.matching.model.bert import BertForNLI


def preprocess_data(data: DataSet, bert_dir):
    """
    preprocess data set to bert-need data set.
    :param data:
    :param bert_dir:
    :return:
    """
    tokenizer = BertTokenizer.from_pretrained(os.path.join(bert_dir, 'vocab.txt'))

    vocab = Vocabulary(padding=None, unknown=None)
    with open(os.path.join(bert_dir, 'vocab.txt')) as f:
        lines = f.readlines()
    vocab_list = []
    for line in lines:
        vocab_list.append(line.strip())
    vocab.add_word_lst(vocab_list)
    vocab.build_vocab()
    vocab.padding = '[PAD]'
    vocab.unknown = '[UNK]'

    for i in range(2):
        data.apply(lambda x: tokenizer.tokenize(" ".join(x[Const.INPUTS(i)])),
                   new_field_name=Const.INPUTS(i))
    data.apply(lambda x: ['[CLS]'] + x[Const.INPUTS(0)] + ['[SEP]'] + x[Const.INPUTS(1)] + ['[SEP]'],
               new_field_name=Const.INPUT)
    data.apply(lambda x: [0] * (len(x[Const.INPUTS(0)]) + 2) + [1] * (len(x[Const.INPUTS(1)]) + 1),
               new_field_name=Const.INPUT_LENS(0))
    data.apply(lambda x: [1] * len(x[Const.INPUT_LENS(0)]), new_field_name=Const.INPUT_LENS(1))

    max_len = 512
    data.apply(lambda x: x[Const.INPUT][: max_len], new_field_name=Const.INPUT)
    data.apply(lambda x: [vocab.to_index(w) for w in x[Const.INPUT]], new_field_name=Const.INPUT)
    data.apply(lambda x: x[Const.INPUT_LENS(0)][: max_len], new_field_name=Const.INPUT_LENS(0))
    data.apply(lambda x: x[Const.INPUT_LENS(1)][: max_len], new_field_name=Const.INPUT_LENS(1))

    target_vocab = Vocabulary(padding=None, unknown=None)
    target_vocab.add_word_lst(['neutral', 'contradiction', 'entailment'])
    target_vocab.build_vocab()
    data.apply(lambda x: target_vocab.to_index(x[Const.TARGET]), new_field_name=Const.TARGET)

    data.set_input(Const.INPUT, Const.INPUT_LENS(0), Const.INPUT_LENS(1), Const.TARGET)
    data.set_target(Const.TARGET)

    return data


bert_dirs = 'path/to/bert/dir'

# load raw data set
train_data = SNLILoader().load('./data/snli/snli_1.0_train.jsonl')
dev_data = SNLILoader().load('./data/snli/snli_1.0_dev.jsonl')
test_data = SNLILoader().load('./data/snli/snli_1.0_test.jsonl')

print('successfully load data sets!')

train_data = preprocess_data(train_data, bert_dirs)
dev_data = preprocess_data(dev_data, bert_dirs)
test_data = preprocess_data(test_data, bert_dirs)

model = BertForNLI(bert_dir=bert_dirs)

trainer = Trainer(
    train_data=train_data,
    model=model,
    optimizer=Adam(lr=2e-5, model_params=model.parameters()),
    batch_size=torch.cuda.device_count() * 12,
    n_epochs=4,
    print_every=-1,
    dev_data=dev_data,
    metrics=AccuracyMetric(),
    metric_key='acc',
    device=[i for i in range(torch.cuda.device_count())],
    check_code_level=-1
)
trainer.train(load_best_model=True)

tester = Tester(
    data=test_data,
    model=model,
    metrics=AccuracyMetric(),
    batch_size=torch.cuda.device_count() * 12,
    device=[i for i in range(torch.cuda.device_count())],
)
tester.test()



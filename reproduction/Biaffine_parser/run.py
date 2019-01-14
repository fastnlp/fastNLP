import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import fastNLP
import torch

from fastNLP.core.trainer import Trainer
from fastNLP.core.instance import Instance
from fastNLP.api.pipeline import Pipeline
from fastNLP.models.biaffine_parser import BiaffineParser, ParserMetric, ParserLoss
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.dataset import DataSet
from fastNLP.core.tester import Tester
from fastNLP.io.config_io import ConfigLoader, ConfigSection
from fastNLP.io.model_io import ModelLoader
from fastNLP.io.embed_loader import EmbedLoader
from fastNLP.io.model_io import ModelSaver
from reproduction.Biaffine_parser.util import ConllxDataLoader, MyDataloader
from fastNLP.api.processor import *

BOS = '<BOS>'
EOS = '<EOS>'
UNK = '<UNK>'
NUM = '<NUM>'
ENG = '<ENG>'

# not in the file's dir
if len(os.path.dirname(__file__)) != 0:
    os.chdir(os.path.dirname(__file__))

def convert(data):
    dataset = DataSet()
    for sample in data:
        word_seq = [BOS] + sample[0]
        pos_seq = [BOS] + sample[1]
        heads = [0] + list(map(int, sample[2]))
        head_tags = [BOS] + sample[3]
        dataset.append(Instance(words=word_seq,
                                pos=pos_seq,
                                gold_heads=heads,
                                arc_true=heads,
                                tags=head_tags))
    return dataset


def load(path):
    data = ConllxDataLoader().load(path)
    return convert(data)


# datadir = "/mnt/c/Me/Dev/release-2.2-st-train-dev-data/ud-treebanks-v2.2/UD_English-EWT"
# datadir = "/home/yfshao/UD_English-EWT"
# train_data_name = "en_ewt-ud-train.conllu"
# dev_data_name = "en_ewt-ud-dev.conllu"
# emb_file_name = '/home/yfshao/glove.6B.100d.txt'
# loader = ConlluDataLoader()

# datadir = '/home/yfshao/workdir/parser-data/'
# train_data_name = "train_ctb5.txt"
# dev_data_name = "dev_ctb5.txt"
# test_data_name = "test_ctb5.txt"

datadir = "/home/yfshao/workdir/ctb7.0/"
train_data_name = "train.conllx"
dev_data_name = "dev.conllx"
test_data_name = "test.conllx"
# emb_file_name = "/home/yfshao/workdir/parser-data/word_OOVthr_30_100v.txt"
emb_file_name = "/home/yfshao/workdir/word_vector/cc.zh.300.vec"

cfgfile = './cfg.cfg'
processed_datadir = './save'

# Config Loader
train_args = ConfigSection()
model_args = ConfigSection()
optim_args = ConfigSection()
ConfigLoader.load_config(cfgfile, {"train": train_args, "model": model_args, "optim": optim_args})
print('trainre Args:', train_args.data)
print('model Args:', model_args.data)
print('optim_args', optim_args.data)


# Pickle Loader
def save_data(dirpath, **kwargs):
    import _pickle
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    for name, data in kwargs.items():
        with open(os.path.join(dirpath, name+'.pkl'), 'wb') as f:
            _pickle.dump(data, f)


def load_data(dirpath):
    import _pickle
    datas = {}
    for f_name in os.listdir(dirpath):
        if not f_name.endswith('.pkl'):
            continue
        name = f_name[:-4]
        with open(os.path.join(dirpath, f_name), 'rb') as f:
            datas[name] = _pickle.load(f)
    return datas

def P2(data, field, length):
    ds = [ins for ins in data if len(ins[field]) >= length]
    data.clear()
    data.extend(ds)
    return ds

def update_v(vocab, data, field):
    data.apply(lambda x: vocab.add_word_lst(x[field]), new_field_name=None)


print('load raw data and preprocess')
# use pretrain embedding
word_v = Vocabulary()
word_v.unknown_label = UNK
pos_v = Vocabulary()
tag_v = Vocabulary(unknown=None, padding=None)
train_data = load(os.path.join(datadir, train_data_name))
dev_data = load(os.path.join(datadir, dev_data_name))
test_data = load(os.path.join(datadir, test_data_name))
print(train_data[0])
num_p = Num2TagProcessor('words', 'words')
for ds in (train_data, dev_data, test_data):
    num_p(ds)

update_v(word_v, train_data, 'words')
update_v(pos_v, train_data, 'pos')
update_v(tag_v, train_data, 'tags')

print('vocab build success {}, {}, {}'.format(len(word_v), len(pos_v), len(tag_v)))
# embed, _ = EmbedLoader.fast_load_embedding(model_args['word_emb_dim'], emb_file_name, word_v)
# print(embed.size())

# Model
model_args['word_vocab_size'] = len(word_v)
model_args['pos_vocab_size'] = len(pos_v)
model_args['num_label'] = len(tag_v)

model = BiaffineParser(**model_args.data)
print(model)

word_idxp = IndexerProcessor(word_v, 'words', 'word_seq')
pos_idxp = IndexerProcessor(pos_v, 'pos', 'pos_seq')
tag_idxp = IndexerProcessor(tag_v, 'tags', 'label_true')
seq_p = SeqLenProcessor('word_seq', 'seq_lens')

set_input_p = SetInputProcessor('word_seq', 'pos_seq', 'seq_lens', flag=True)
set_target_p = SetTargetProcessor('arc_true', 'label_true', 'seq_lens', flag=True)

label_toword_p = Index2WordProcessor(vocab=tag_v, field_name='label_pred', new_added_field_name='label_pred_seq')

for ds in (train_data, dev_data, test_data):
    word_idxp(ds)
    pos_idxp(ds)
    tag_idxp(ds)
    seq_p(ds)
    set_input_p(ds)
    set_target_p(ds)

if train_args['use_golden_train']:
    train_data.set_input('gold_heads', flag=True)
train_args.data.pop('use_golden_train')
ignore_label = pos_v['punct']

print(test_data[0])
print('train len {}'.format(len(train_data)))
print('dev len {}'.format(len(dev_data)))
print('test len {}'.format(len(test_data)))



def train(path):
    # test saving pipeline
    save_pipe(path)

    # Trainer
    trainer = Trainer(model=model, train_data=train_data, dev_data=dev_data,
                      loss=ParserLoss(), metrics=ParserMetric(), metric_key='UAS',
                      **train_args.data,
                      optimizer=fastNLP.Adam(**optim_args.data),
                      save_path=path)

    # model.word_embedding = torch.nn.Embedding.from_pretrained(embed, freeze=False)
    model.word_embedding.padding_idx = word_v.padding_idx
    model.word_embedding.weight.data[word_v.padding_idx].fill_(0)
    model.pos_embedding.padding_idx = pos_v.padding_idx
    model.pos_embedding.weight.data[pos_v.padding_idx].fill_(0)

    # try:
    #     ModelLoader.load_pytorch(model, "./save/saved_model.pkl")
    #     print('model parameter loaded!')
    # except Exception as _:
    #     print("No saved model. Continue.")
    #     pass

    # Start training
    trainer.train()
    print("Training finished!")

    # save pipeline
    save_pipe(path)
    print('pipe saved')

def save_pipe(path):
    pipe = Pipeline(processors=[num_p, word_idxp, pos_idxp, seq_p, set_input_p])
    pipe.add_processor(ModelProcessor(model=model, batch_size=32))
    pipe.add_processor(label_toword_p)
    os.makedirs(path, exist_ok=True)
    torch.save({'pipeline': pipe}, os.path.join(path, 'pipe.pkl'))


def test(path):
    # Tester
    tester = Tester(**test_args.data)

    # Model
    model = BiaffineParser(**model_args.data)
    model.eval()
    try:
        ModelLoader.load_pytorch(model, path)
        print('model parameter loaded!')
    except Exception as _:
        print("No saved model. Abort test.")
        raise

    # Start training
    print("Testing Train data")
    tester.test(model, train_data)
    print("Testing Dev data")
    tester.test(model, dev_data)
    print("Testing Test data")
    tester.test(model, test_data)

def build_pipe(parser_pipe_path):
    parser_pipe = torch.load(parser_pipe_path)




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run a chinese word segmentation model')
    parser.add_argument('--mode', help='set the model\'s model', choices=['train', 'test', 'infer', 'save'])
    parser.add_argument('--path', type=str, default='')
    # parser.add_argument('--dst', type=str, default='')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args.path)
    elif args.mode == 'test':
        test(args.path)
    elif args.mode == 'infer':
        pass
    # elif args.mode == 'save':
    #     print(f'save model from {args.path} to {args.dst}')
    #     save_model(args.path, args.dst)
    #     load_path = os.path.dirname(args.dst)
    #     print(f'save pipeline in {load_path}')
    #     build(load_path)
    else:
        print('no mode specified for model!')
        parser.print_help()

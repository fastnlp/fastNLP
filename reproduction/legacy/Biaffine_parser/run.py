import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import fastNLP

from fastNLP.core.trainer import Trainer
from fastNLP.core.instance import Instance
from fastNLP.api.pipeline import Pipeline
from fastNLP.models.biaffine_parser import BiaffineParser, ParserMetric, ParserLoss
from fastNLP.core.tester import Tester
from fastNLP.io.config_io import ConfigLoader, ConfigSection
from fastNLP.io.model_io import ModelLoader
from fastNLP.io.dataset_loader import ConllxDataLoader
from fastNLP.api.processor import *
from fastNLP.io.embed_loader import EmbedLoader
from fastNLP.core.callback import Callback

BOS = '<BOS>'
EOS = '<EOS>'
UNK = '<UNK>'
PAD = '<PAD>'
NUM = '<NUM>'
ENG = '<ENG>'

# not in the file's dir
if len(os.path.dirname(__file__)) != 0:
    os.chdir(os.path.dirname(__file__))

def convert(data):
    dataset = DataSet()
    for sample in data:
        word_seq = [BOS] + sample['words']
        pos_seq = [BOS] + sample['pos_tags']
        heads = [0] + sample['heads']
        head_tags = [BOS] + sample['labels']
        dataset.append(Instance(raw_words=word_seq,
                                pos=pos_seq,
                                gold_heads=heads,
                                arc_true=heads,
                                tags=head_tags))
    return dataset


def load(path):
    data = ConllxDataLoader().load(path)
    return convert(data)


datadir = "/remote-home/yfshao/workdir/ctb9.0/"
train_data_name = "train.conllx"
dev_data_name = "dev.conllx"
test_data_name = "test.conllx"
emb_file_name = "/remote-home/yfshao/workdir/word_vector/cc.zh.300.vec"

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


# use pretrain embedding
word_v = Vocabulary(unknown=UNK, padding=PAD)
pos_v = Vocabulary(unknown=None, padding=PAD)
tag_v = Vocabulary(unknown=None, padding=None)
train_data = load(os.path.join(datadir, train_data_name))
dev_data = load(os.path.join(datadir, dev_data_name))
test_data = load(os.path.join(datadir, test_data_name))
print('load raw data and preprocess')

num_p = Num2TagProcessor(tag=NUM, field_name='raw_words', new_added_field_name='words')
for ds in (train_data, dev_data, test_data):
    num_p(ds)
update_v(word_v, train_data, 'words')
update_v(pos_v, train_data, 'pos')
update_v(tag_v, train_data, 'tags')

print('vocab build success {}, {}, {}'.format(len(word_v), len(pos_v), len(tag_v)))

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

print(test_data[0])
print('train len {}'.format(len(train_data)))
print('dev len {}'.format(len(dev_data)))
print('test len {}'.format(len(test_data)))


def train(path):
    # test saving pipeline
    save_pipe(path)
    embed = EmbedLoader.load_with_vocab(emb_file_name, word_v)
    embed = torch.tensor(embed, dtype=torch.float32)

    # embed = EmbedLoader.fast_load_embedding(emb_dim=model_args['word_emb_dim'], emb_file=emb_file_name, vocab=word_v)
    # embed = torch.tensor(embed, dtype=torch.float32)
    # model.word_embedding = torch.nn.Embedding.from_pretrained(embed, freeze=True)
    model.word_embedding.padding_idx = word_v.padding_idx
    model.word_embedding.weight.data[word_v.padding_idx].fill_(0)
    model.pos_embedding.padding_idx = pos_v.padding_idx
    model.pos_embedding.weight.data[pos_v.padding_idx].fill_(0)

    class MyCallback(Callback):
        def on_step_end(self, optimizer):
            step = self.trainer.step
            # learning rate decay
            if step > 0 and step % 1000 == 0:
                for pg in optimizer.param_groups:
                    pg['lr'] *= 0.93
                print('decay lr to {}'.format([pg['lr'] for pg in optimizer.param_groups]))

            if step == 3000:
                # start training embedding
                print('start training embedding at {}'.format(step))
                model = self.trainer.model
                for m in model.modules():
                    if isinstance(m, torch.nn.Embedding):
                        m.weight.requires_grad = True

    # Trainer
    trainer = Trainer(train_data=train_data, model=model, optimizer=fastNLP.Adam(**optim_args.data), loss=ParserLoss(),
                      dev_data=dev_data, metrics=ParserMetric(), metric_key='UAS', save_path=path,
                      callbacks=[MyCallback()])

    # Start training
    try:
        trainer.train()
        print("Training finished!")
    finally:
        # save pipeline
        save_pipe(path)
        print('pipe saved')

def save_pipe(path):
    pipe = Pipeline(processors=[num_p, word_idxp, pos_idxp, seq_p, set_input_p])
    pipe.add_processor(ModelProcessor(model=model, batch_size=32))
    pipe.add_processor(label_toword_p)
    os.makedirs(path, exist_ok=True)
    torch.save({'pipeline': pipe,
                'names':['num word_idx pos_idx seq set_input model tag_to_word'.split()],
                }, os.path.join(path, 'pipe.pkl'))


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run a chinese word segmentation model')
    parser.add_argument('--mode', help='set the model\'s model', choices=['train', 'test', 'infer'])
    parser.add_argument('--path', type=str, default='')
    # parser.add_argument('--dst', type=str, default='')
    args = parser.parse_args()
    if args.mode == 'train':
        train(args.path)
    elif args.mode == 'test':
        test(args.path)
    elif args.mode == 'infer':
        pass
    else:
        print('no mode specified for model!')
        parser.print_help()

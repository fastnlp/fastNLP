import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from collections import defaultdict
import math
import torch

from fastNLP.core.trainer import Trainer
from fastNLP.core.instance import Instance
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.dataset import DataSet
from fastNLP.core.batch import Batch
from fastNLP.core.sampler import SequentialSampler
from fastNLP.core.field import TextField, SeqLabelField
from fastNLP.core.tester import Tester
from fastNLP.loader.config_loader import ConfigLoader, ConfigSection
from fastNLP.loader.model_loader import ModelLoader
from fastNLP.loader.embed_loader import EmbedLoader
from fastNLP.models.biaffine_parser import BiaffineParser
from fastNLP.saver.model_saver import ModelSaver

# not in the file's dir
if len(os.path.dirname(__file__)) != 0:
    os.chdir(os.path.dirname(__file__))

class ConlluDataLoader(object):
    def load(self, path):
        datalist = []
        with open(path, 'r', encoding='utf-8') as f:
            sample = []
            for line in f:
                if line.startswith('\n'):
                    datalist.append(sample)
                    sample = []
                elif line.startswith('#'):
                    continue
                else:
                    sample.append(line.split('\t'))
            if len(sample) > 0:
                datalist.append(sample)

        ds = DataSet(name='conll')
        for sample in datalist:
            # print(sample)
            res = self.get_one(sample)
            ds.append(Instance(word_seq=TextField(res[0], is_target=False),
                               pos_seq=TextField(res[1], is_target=False),
                               head_indices=SeqLabelField(res[2], is_target=True),
                               head_labels=TextField(res[3], is_target=True)))

        return ds

    def get_one(self, sample):
        text = ['<root>']
        pos_tags = ['<root>']
        heads = [0]
        head_tags = ['root']
        for w in sample:
            t1, t2, t3, t4 = w[1], w[3], w[6], w[7]
            if t3 == '_':
                continue
            text.append(t1)
            pos_tags.append(t2)
            heads.append(int(t3))
            head_tags.append(t4)
        return (text, pos_tags, heads, head_tags)

class CTBDataLoader(object):
    def load(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = self.parse(lines)
        return self.convert(data)

    def parse(self, lines):
        """
            [
                [word], [pos], [head_index], [head_tag]
            ]
        """
        sample = []
        data = []
        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) == 0 or i+1 == len(lines):
                data.append(list(map(list, zip(*sample))))
                sample = []
            else:
                sample.append(line.split())
        return data

    def convert(self, data):
        dataset = DataSet()
        for sample in data:
            word_seq = ["<ROOT>"] + sample[0]
            pos_seq = ["<ROOT>"] + sample[1]
            heads = [0] + list(map(int, sample[2]))
            head_tags = ["ROOT"] + sample[3]
            dataset.append(Instance(word_seq=TextField(word_seq, is_target=False),
                                    pos_seq=TextField(pos_seq, is_target=False),
                                    head_indices=SeqLabelField(heads, is_target=True),
                                    head_labels=TextField(head_tags, is_target=True)))
        return dataset

# datadir = "/mnt/c/Me/Dev/release-2.2-st-train-dev-data/ud-treebanks-v2.2/UD_English-EWT"
# datadir = "/home/yfshao/UD_English-EWT"
# train_data_name = "en_ewt-ud-train.conllu"
# dev_data_name = "en_ewt-ud-dev.conllu"
# emb_file_name = '/home/yfshao/glove.6B.100d.txt'
# loader = ConlluDataLoader()

datadir = "/home/yfshao/parser-data"
train_data_name = "train_ctb5.txt"
dev_data_name = "dev_ctb5.txt"
emb_file_name = "/home/yfshao/parser-data/word_OOVthr_30_100v.txt"
loader = CTBDataLoader()

cfgfile = './cfg.cfg'
processed_datadir = './save'

# Config Loader
train_args = ConfigSection()
test_args = ConfigSection()
model_args = ConfigSection()
optim_args = ConfigSection()
ConfigLoader.load_config(cfgfile, {"train": train_args, "test": test_args, "model": model_args, "optim": optim_args})

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

class MyTester(object):
    def __init__(self, batch_size, use_cuda=False, **kwagrs):
        self.batch_size = batch_size
        self.use_cuda = use_cuda

    def test(self, model, dataset):
        self.model = model.cuda() if self.use_cuda else model
        self.model.eval()
        batchiter = Batch(dataset, self.batch_size, SequentialSampler(), self.use_cuda)
        eval_res = defaultdict(list)
        i = 0
        for batch_x, batch_y in batchiter:
            with torch.no_grad():
                pred_y = self.model(**batch_x)
                eval_one = self.model.evaluate(**pred_y, **batch_y)
            i += self.batch_size
            for eval_name, tensor in eval_one.items():
                eval_res[eval_name].append(tensor)
        tmp = {}
        for eval_name, tensorlist in eval_res.items():
            tmp[eval_name] = torch.cat(tensorlist, dim=0)

        self.res = self.model.metrics(**tmp)
        print(self.show_metrics())

    def show_metrics(self):
        s = ""
        for name, val in self.res.items():
            s += '{}: {:.2f}\t'.format(name, val)
        return s


try:
    data_dict = load_data(processed_datadir)
    word_v = data_dict['word_v']
    pos_v = data_dict['pos_v']
    tag_v = data_dict['tag_v']
    train_data = data_dict['train_data']
    dev_data = data_dict['dev_data']
    print('use saved pickles')

except Exception as _:
    print('load raw data and preprocess')
    word_v = Vocabulary(need_default=True, min_freq=2)
    pos_v = Vocabulary(need_default=True)
    tag_v = Vocabulary(need_default=False)
    train_data = loader.load(os.path.join(datadir, train_data_name))
    dev_data = loader.load(os.path.join(datadir, dev_data_name))
    train_data.update_vocab(word_seq=word_v, pos_seq=pos_v, head_labels=tag_v)
    save_data(processed_datadir, word_v=word_v, pos_v=pos_v, tag_v=tag_v, train_data=train_data, dev_data=dev_data)

train_data.index_field("word_seq", word_v).index_field("pos_seq", pos_v).index_field("head_labels", tag_v)
dev_data.index_field("word_seq", word_v).index_field("pos_seq", pos_v).index_field("head_labels", tag_v)
train_data.set_origin_len("word_seq")
dev_data.set_origin_len("word_seq")

print(train_data[:3])
print(len(train_data))
print(len(dev_data))
ep = train_args['epochs']
train_args['epochs'] =  math.ceil(50000.0 / len(train_data) * train_args['batch_size']) if ep <= 0 else ep
model_args['word_vocab_size'] = len(word_v)
model_args['pos_vocab_size'] = len(pos_v)
model_args['num_label'] = len(tag_v)


def train():
    # Trainer
    trainer = Trainer(**train_args.data)

    def _define_optim(obj):
        obj._optimizer = torch.optim.Adam(obj._model.parameters(), **optim_args.data)
        obj._scheduler = torch.optim.lr_scheduler.LambdaLR(obj._optimizer, lambda ep: .75 ** (ep / 5e4))

    def _update(obj):
        obj._scheduler.step()
        obj._optimizer.step()

    trainer.define_optimizer = lambda: _define_optim(trainer)
    trainer.update = lambda: _update(trainer)
    trainer.get_loss = lambda predict, truth: trainer._loss_func(**predict, **truth)
    trainer._create_validator = lambda x: MyTester(**test_args.data)

    # Model
    model = BiaffineParser(**model_args.data)

    # use pretrain embedding
    word_v.unknown_label = "<OOV>"
    embed, _ = EmbedLoader.load_embedding(model_args['word_emb_dim'], emb_file_name, 'glove', word_v, os.path.join(processed_datadir, 'word_emb.pkl'))
    model.word_embedding = torch.nn.Embedding.from_pretrained(embed, freeze=False)
    model.word_embedding.padding_idx = word_v.padding_idx
    model.word_embedding.weight.data[word_v.padding_idx].fill_(0)
    model.pos_embedding.padding_idx = pos_v.padding_idx
    model.pos_embedding.weight.data[pos_v.padding_idx].fill_(0)

    try:
        ModelLoader.load_pytorch(model, "./save/saved_model.pkl")
        print('model parameter loaded!')
    except Exception as _:
        print("No saved model. Continue.")
        pass

    # Start training
    trainer.train(model, train_data, dev_data)
    print("Training finished!")

    # Saver
    saver = ModelSaver("./save/saved_model.pkl")
    saver.save_pytorch(model)
    print("Model saved!")


def test():
    # Tester
    tester = MyTester(**test_args.data)

    # Model
    model = BiaffineParser(**model_args.data)

    try:
        ModelLoader.load_pytorch(model, "./save/saved_model.pkl")
        print('model parameter loaded!')
    except Exception as _:
        print("No saved model. Abort test.")
        raise

    # Start training
    tester.test(model, dev_data)
    print(tester.show_metrics())
    print("Testing finished!")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run a chinese word segmentation model')
    parser.add_argument('--mode', help='set the model\'s model', choices=['train', 'test', 'infer'])
    args = parser.parse_args()
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode == 'infer':
        infer()
    else:
        print('no mode specified for model!')
        parser.print_help()

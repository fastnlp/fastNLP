import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import re

from fastNLP.core.trainer import Trainer
from fastNLP.core.metrics import Evaluator
from fastNLP.core.instance import Instance
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.dataset import DataSet
from fastNLP.core.field import TextField, SeqLabelField
from fastNLP.core.tester import Tester
from fastNLP.io.config_loader import ConfigLoader, ConfigSection
from fastNLP.io.model_loader import ModelLoader
from fastNLP.io.embed_loader import EmbedLoader
from fastNLP.models.biaffine_parser import BiaffineParser
from fastNLP.io.model_saver import ModelSaver

BOS = '<BOS>'
EOS = '<EOS>'
UNK = '<OOV>'
NUM = '<NUM>'
ENG = '<ENG>'

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
        text = []
        pos_tags = []
        heads = []
        head_tags = []
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
            word_seq = [BOS] + sample[0] + [EOS]
            pos_seq = [BOS] + sample[1] + [EOS]
            heads = [0] + list(map(int, sample[2])) + [0]
            head_tags = [BOS] + sample[3] + [EOS]
            dataset.append(Instance(word_seq=TextField(word_seq, is_target=False),
                                    pos_seq=TextField(pos_seq, is_target=False),
                                    gold_heads=SeqLabelField(heads, is_target=False),
                                    head_indices=SeqLabelField(heads, is_target=True),
                                    head_labels=TextField(head_tags, is_target=True)))
        return dataset

# datadir = "/mnt/c/Me/Dev/release-2.2-st-train-dev-data/ud-treebanks-v2.2/UD_English-EWT"
# datadir = "/home/yfshao/UD_English-EWT"
# train_data_name = "en_ewt-ud-train.conllu"
# dev_data_name = "en_ewt-ud-dev.conllu"
# emb_file_name = '/home/yfshao/glove.6B.100d.txt'
# loader = ConlluDataLoader()

datadir = '/home/yfshao/workdir/parser-data/'
train_data_name = "train_ctb5.txt"
dev_data_name = "dev_ctb5.txt"
test_data_name = "test_ctb5.txt"
emb_file_name = "/home/yfshao/workdir/parser-data/word_OOVthr_30_100v.txt"
# emb_file_name = "/home/yfshao/workdir/word_vector/cc.zh.300.vec"
loader = CTBDataLoader()

cfgfile = './cfg.cfg'
processed_datadir = './save'

# Config Loader
train_args = ConfigSection()
test_args = ConfigSection()
model_args = ConfigSection()
optim_args = ConfigSection()
ConfigLoader.load_config(cfgfile, {"train": train_args, "test": test_args, "model": model_args, "optim": optim_args})
print('trainre Args:', train_args.data)
print('test Args:', test_args.data)
print('optim Args:', optim_args.data)


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
    ds = [ins for ins in data if ins[field].get_length() >= length]
    data.clear()
    data.extend(ds)
    return ds

def P1(data, field):
    def reeng(w):
        return w if w == BOS or w == EOS or re.search(r'^([a-zA-Z]+[\.\-]*)+$', w) is None else ENG
    def renum(w):
        return w if re.search(r'^[0-9]+\.?[0-9]*$', w) is None else NUM
    for ins in data:
        ori = ins[field].contents()
        s = list(map(renum, map(reeng, ori)))
        if s != ori:
            # print(ori)
            # print(s)
            # print()
            ins[field] = ins[field].new(s)
    return data

class ParserEvaluator(Evaluator):
    def __init__(self, ignore_label):
        super(ParserEvaluator, self).__init__()
        self.ignore = ignore_label

    def __call__(self, predict_list, truth_list):
        head_all, label_all, total_all = 0, 0, 0
        for pred, truth in zip(predict_list, truth_list):
            head, label, total = self.evaluate(**pred, **truth)
            head_all += head
            label_all += label
            total_all += total

        return {'UAS': head_all*1.0 / total_all, 'LAS': label_all*1.0 / total_all}

    def evaluate(self, head_pred, label_pred, head_indices, head_labels, seq_mask, **_):
        """
        Evaluate the performance of prediction.

        :return : performance results.
            head_pred_corrct: number of correct predicted heads.
            label_pred_correct: number of correct predicted labels.
            total_tokens: number of predicted tokens
        """
        seq_mask *= (head_labels != self.ignore).long()
        head_pred_correct = (head_pred == head_indices).long() * seq_mask
        _, label_preds = torch.max(label_pred, dim=2)
        label_pred_correct = (label_preds == head_labels).long() * head_pred_correct
        return head_pred_correct.sum().item(), label_pred_correct.sum().item(), seq_mask.sum().item()

try:
    data_dict = load_data(processed_datadir)
    word_v = data_dict['word_v']
    pos_v = data_dict['pos_v']
    tag_v = data_dict['tag_v']
    train_data = data_dict['train_data']
    dev_data = data_dict['dev_data']
    test_data = data_dict['test_data']
    print('use saved pickles')

except Exception as _:
    print('load raw data and preprocess')
    # use pretrain embedding
    word_v = Vocabulary(need_default=True, min_freq=2)
    word_v.unknown_label = UNK
    pos_v = Vocabulary(need_default=True)
    tag_v = Vocabulary(need_default=False)
    train_data = loader.load(os.path.join(datadir, train_data_name))
    dev_data = loader.load(os.path.join(datadir, dev_data_name))
    test_data = loader.load(os.path.join(datadir, test_data_name))
    train_data.update_vocab(word_seq=word_v, pos_seq=pos_v, head_labels=tag_v)
    datasets = (train_data, dev_data, test_data)
    save_data(processed_datadir, word_v=word_v, pos_v=pos_v, tag_v=tag_v, train_data=train_data, dev_data=dev_data, test_data=test_data)

embed, _ = EmbedLoader.load_embedding(model_args['word_emb_dim'], emb_file_name, 'glove', word_v, os.path.join(processed_datadir, 'word_emb.pkl'))

print(len(word_v))
print(embed.size())

# Model
model_args['word_vocab_size'] = len(word_v)
model_args['pos_vocab_size'] = len(pos_v)
model_args['num_label'] = len(tag_v)

model = BiaffineParser(**model_args.data)
model.reset_parameters()
datasets = (train_data, dev_data, test_data)
for ds in datasets:
    ds.index_field("word_seq", word_v).index_field("pos_seq", pos_v).index_field("head_labels", tag_v)
    ds.set_origin_len('word_seq')
if train_args['use_golden_train']:
    train_data.set_target(gold_heads=False)
else:
    train_data.set_target(gold_heads=None)
train_args.data.pop('use_golden_train')
ignore_label = pos_v['P']

print(test_data[0])
print(len(train_data))
print(len(dev_data))
print(len(test_data))



def train(path):
    # Trainer
    trainer = Trainer(**train_args.data)

    def _define_optim(obj):
        lr = optim_args.data['lr']
        embed_params = set(obj._model.word_embedding.parameters())
        decay_params = set(obj._model.arc_predictor.parameters()) | set(obj._model.label_predictor.parameters())
        params = [p for p in obj._model.parameters() if p not in decay_params and p not in embed_params]
        obj._optimizer = torch.optim.Adam([
            {'params': list(embed_params), 'lr':lr*0.1},
            {'params': list(decay_params), **optim_args.data},
            {'params': params}
            ], lr=lr, betas=(0.9, 0.9))
        obj._scheduler = torch.optim.lr_scheduler.LambdaLR(obj._optimizer, lambda ep: max(.75 ** (ep / 5e4), 0.05))

    def _update(obj):
        # torch.nn.utils.clip_grad_norm_(obj._model.parameters(), 5.0)
        obj._scheduler.step()
        obj._optimizer.step()

    trainer.define_optimizer = lambda: _define_optim(trainer)
    trainer.update = lambda: _update(trainer)
    trainer.set_validator(Tester(**test_args.data, evaluator=ParserEvaluator(ignore_label)))

    model.word_embedding = torch.nn.Embedding.from_pretrained(embed, freeze=False)
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
    trainer.train(model, train_data, dev_data)
    print("Training finished!")

    # Saver
    saver = ModelSaver("./save/saved_model.pkl")
    saver.save_pytorch(model)
    print("Model saved!")


def test(path):
    # Tester
    tester = Tester(**test_args.data, evaluator=ParserEvaluator(ignore_label))

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

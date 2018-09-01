import _pickle
import os

import numpy as np
import torch

from fastNLP.core.preprocess import SeqLabelPreprocess
from fastNLP.core.tester import SeqLabelTester
from fastNLP.core.trainer import SeqLabelTrainer
from fastNLP.models.sequence_modeling import AdvSeqLabel


class MyNERTrainer(SeqLabelTrainer):
    def __init__(self, train_args):
        super(MyNERTrainer, self).__init__(train_args)
        self.scheduler = None

    def define_optimizer(self):
        """
        override
        :return:
        """
        self.optimizer = torch.optim.Adam(self._model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.5)

    def update(self):
        """
        override
        :return:
        """
        self.optimizer.step()
        self.scheduler.step()

    def _create_validator(self, valid_args):
        return MyNERTester(valid_args)

    def best_eval_result(self, validator):
        accuracy = validator.metrics()
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            return True
        else:
            return False


class MyNERTester(SeqLabelTester):
    def __init__(self, test_args):
        super(MyNERTester, self).__init__(test_args)

    def _evaluate(self, prediction, batch_y, seq_len):
        """
        :param prediction: [batch_size, seq_len, num_classes]
        :param batch_y: [batch_size, seq_len]
        :param seq_len: [batch_size]
        :return:
        """
        summ = 0
        correct = 0
        _, indices = torch.max(prediction, 2)
        for p, y, l in zip(indices, batch_y, seq_len):
            summ += l
            correct += np.sum(p[:l].cpu().numpy() == y[:l].cpu().numpy())
        return float(correct / summ)

    def evaluate(self, predict, truth):
        return self._evaluate(predict, truth, self.seq_len)

    def metrics(self):
        return np.mean(self.eval_history)

    def show_metrics(self):
        return "dev accuracy={:.2f}".format(float(self.metrics()))


def embedding_process(emb_file, word_dict, emb_dim, emb_pkl):
    if os.path.exists(emb_pkl):
        with open(emb_pkl, "rb") as f:
            embedding_np = _pickle.load(f)
        return embedding_np
    with open(emb_file, "r", encoding="utf-8") as f:
        embedding_np = np.random.uniform(-1, 1, size=(len(word_dict), emb_dim))
        for line in f:
            line = line.strip().split()
            if len(line) != emb_dim + 1:
                continue
            if line[0] in word_dict:
                embedding_np[word_dict[line[0]]] = [float(i) for i in line[1:]]
    with open(emb_pkl, "wb") as f:
        _pickle.dump(embedding_np, f)
    return embedding_np


def data_load(data_file):
    with open(data_file, "r", encoding="utf-8") as f:
        all_data = []
        sent = []
        label = []
        for line in f:
            line = line.strip().split()

            if not len(line) <= 1:
                sent.append(line[0])
                label.append(line[1])
            else:
                all_data.append([sent, label])
                sent = []
                label = []
    return all_data


data_path = "data_for_tests/people.txt"
pick_path = "data_for_tests/"
emb_path = "data_for_tests/emb50.txt"
save_path = "data_for_tests/"
if __name__ == "__main__":
    data = data_load(data_path)
    preprocess = SeqLabelPreprocess()
    data_train, data_dev = preprocess.run(data, pickle_path=pick_path, train_dev_split=0.3)
    # emb = embedding_process(emb_path, p.word2index, 50, os.path.join(pick_path, "embedding.pkl"))
    emb = None
    args = {"epochs": 20,
            "batch_size": 1,
            "pickle_path": pick_path,
            "validate": True,
            "save_best_dev": True,
            "model_saved_path": save_path,
            "use_cuda": True,

            "vocab_size": preprocess.vocab_size,
            "num_classes": preprocess.num_classes,
            "word_emb_dim": 50,
            "rnn_hidden_units": 100
            }
    # emb = torch.Tensor(emb).float().cuda()
    networks = AdvSeqLabel(args, emb)
    trainer = MyNERTrainer(args)
    trainer.train(networks, data_train, data_dev)
    print("Training finished!")

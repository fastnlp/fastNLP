import _pickle

import numpy as np
import torch

from .action import Action
from .action import RandomSampler, Batchifier
from .tester import POSTester
from ..modules.utils import seq_mask
import os
import random
class BaseTrainer(Action):
    """Base trainer for all trainers.
        Trainer receives a model and data, and then performs training.

        Subclasses must implement the following abstract methods:
        - prepare_input
        - mode
        - define_optimizer
        - data_forward
        - grad_backward
        - get_loss
    """

    def __init__(self, train_args):
        """
        :param train_args: dict of (key, value)

        The base trainer requires the following keys:
        - epochs: int, the number of epochs in training
        - validate: bool, whether or not to validate on dev set
        - batch_size: int
        - pickle_path: str, the path to pickle files for pre-processing
        """
        super(BaseTrainer, self).__init__()
        self.n_epochs = train_args["epochs"]
        self.validate = train_args["validate"]
        self.batch_size = train_args["batch_size"]
        self.pickle_path = train_args["pickle_path"]
        self.use_gpu=train_args["use_gpu"]
        self.model = None

        self.iterator = None
        self.loss_func = None
        self.optimizer = None

    def train(self, network,use_gpu=False,label_dict=None):
        """General Training Steps
        :param network: a model

        The method is framework independent.
        Work by calling the following methods:
            - prepare_input
            - mode
            - define_optimizer
            - data_forward
            - get_loss
            - grad_backward
            - update
        Subclasses must implement these methods with a specific framework.
        """
        # prepare model and data

        self.model = network
        self.model.load_state_dict(torch.load("save/model91.8.pkl"))
        if use_gpu:
            self.model=self.model.cuda()
        data_train, data_dev= self.prepare_input(self.pickle_path)

        # define tester over dev data
        valid_args = {"save_output": True, "validate_in_training": True, "save_dev_input": True,
                      "save_loss": True, "batch_size": self.batch_size, "pickle_path": self.pickle_path}
        #validator = POSTester(valid_args)

        # main training epochs
        iterations = len(data_train) // self.batch_size
        print("train number is",iterations)
        optimizer = self.define_optimizer()
        for epoch in range(self.n_epochs):

            # turn on network training mode; define optimizer; prepare batch iterator
            #self.mode(test=False)

            self.iterator = iter(Batchifier(RandomSampler(data_train), self.batch_size, drop_last=True))

            # training iterations in one epoch
            for step in range(iterations):
                batch_x, batch_y,seq_len = self.batchify(data_train)
                # print(batch_x)
                # print(batch_y)
                batch_x=np.array(batch_x,np.int32)
                batch_y=np.array(batch_y,np.int32)
                seq_len=np.array(seq_len,np.int32)
                prediction = self.data_forward(network, batch_x)

                loss = self.get_loss(prediction, batch_y,seq_mask(seq_len,np.shape(batch_x)[1]))
                if step%200==0:
                    print("epoch :",epoch,"step {}/{}:".format(step,iterations),"loss :",loss)
                optimizer.zero_grad()
                self.grad_backward(loss)
                optimizer.step()
                #self.update()

            if self.validate:
                self.dev(data_dev,label_dict)
                # if data_dev is None:
                #     raise RuntimeError("No validation data provided.")
                # validator.test(network)
                # print("[epoch {}] dev loss={:.2f}".format(epoch, validator.matrices()))

        # finish training

    def dev(self, data_dev,label_dict=None):

        if data_dev is None:
            raise RuntimeError("No validation data provided.")
        best_acc = 0
        iterations = len(data_dev) // self.batch_size
        print("dev number is",iterations)
        self.iterator = iter(Batchifier(RandomSampler(data_dev), self.batch_size, drop_last=True))
        correct = 0
        total = 0
        per=0
        la=0
        real=0
        for step in range(iterations):
            batch_x, batch_y, seq_len = self.batchify(data_dev)
            y=batch_y
            s=seq_len
            batch_x = np.array(batch_x, np.int32)
            batch_y = np.array(batch_y, np.int32)
            seq_len = np.array(seq_len, np.int32)
            prediction = self.model.result(self.model(batch_x),seq_mask(seq_len,np.shape(batch_x)[1]))

            # if not isinstance(prediction, list):
            #     print(prediction)

            for i in range(len(y)):


                p=prediction[i]
                for j in p:
                    if label_dict[j][2:]=="ns" or label_dict[j][2:]=="nr" or label_dict[j][2:]=="t" or label_dict[j][2:]=="nz" or label_dict[j][2:]=="nt":
                        per+=1

                l=y[i][:s[i]]
                for j in l:
                    if label_dict[j][2:] == "ns" or label_dict[j][2:] == "nr" or label_dict[j][2:] == "t" or label_dict[j][2:] == "nz" or label_dict[j][2:] == "nt":
                        la+=1

                for j,k in zip(p,l):
                    if label_dict[j][2:] == "ns" or label_dict[j][2:] == "nr" or label_dict[j][2:] == "t" or label_dict[j][2:] == "nz" or label_dict[j][2:] == "nt":
                        if j==k:
                            real+=1


                total += len(p)
                correct += np.sum(p==l)
        percision=real/per
        recall=real/la
        f1=percision*recall*2/(percision+recall)
        acc = correct * 100 / total
        print("dev acc is: %3.2f%% , percision is: %3.2f%% , recall is: %3.2f%% , f1 is: %3.2f%%" % (acc,percision,recall,f1))
        if f1 > best_acc:
            best_acc = f1
            torch.save(self.model.state_dict(),"save/model{}.pkl".format(round(acc,1)))
            print("have saved")
        return best_acc

    def prepare_input(self, data_path):
        """
            To do: Load pkl files of train/dev/test and embedding
        """
        data_train = _pickle.load(open(data_path + "/data_train.pkl", "rb"))
        if os.path.exists(data_path + "/data_dev.pkl"):
            data_dev = _pickle.load(open(data_path + "/data_dev.pkl", "rb"))
        else:
            random.shuffle(data_train)
            data_dev=data_train[-int(len(data_train)*0.02):]
            data_train=data_train[:-int(len(data_train)*0.02)]
        #data_test = _pickle.load(open(data_path + "/data_test.pkl", "rb"))
        #embedding = _pickle.load(open(data_path + "/embedding.pkl", "rb"))
        return data_train, data_dev

    def mode(self, test=False):
        """
        Tell the network to be trained or not.
        :param test: bool
        """
        raise NotImplementedError

    def define_optimizer(self):
        """
        Define framework-specific optimizer specified by the models.
        """
        return torch.optim.SGD(self.model.parameters(), lr=0.0005, momentum=0.9)

    def update(self):
        """
        Perform weight update on a model.

        For PyTorch, just call optimizer to update.
        """
        raise NotImplementedError

    def data_forward(self, network, x):
        """
        Forward pass of the data.
        :param network: a model
        :param x: input feature matrix and label vector
        :return: output by the models

        For PyTorch, just do "network(*x)"
        """
        return network(x)

    def grad_backward(self, loss):
        """
        Compute gradient with link rules.
        :param loss: a scalar where back-prop starts

        For PyTorch, just do "loss.backward()"
        """
        return loss.backward()

    def get_loss(self, predict, truth , mask=None):
        """
        Compute loss given prediction and ground truth.
        :param predict: prediction label vector
        :param truth: ground truth label vector
        :return: a scalar
        """
        if self.loss_func is None:
            if hasattr(self.model, "loss"):
                self.loss_func = self.model.loss
            else:
                self.define_loss()
        return self.loss_func(predict, truth , mask)

    def define_loss(self):
        """
            Assign an instance of loss function to self.loss_func
            E.g. self.loss_func = nn.CrossEntropyLoss()
        """
        raise NotImplementedError

    def batchify(self, data):
        """
        1. Perform batching from data and produce a batch of training data.
        2. Add padding.
        :param data: list. Each entry is a sample, which is also a list of features and label(s).
            E.g.
                [
                    [[word_11, word_12, word_13], [label_11. label_12]],  # sample 1
                    [[word_21, word_22, word_23], [label_21. label_22]],  # sample 2
                    ...
                ]
        :return batch_x: list. Each entry is a list of features of a sample. [batch_size, max_len]
                 batch_y: list. Each entry is a list of labels of a sample.  [batch_size, num_labels]
        """
        indices = next(self.iterator)
        batch = [data[idx] for idx in indices]
        batch_x = [sample[0] for sample in batch]
        batch_y = [sample[1] for sample in batch]
        seq_len = [len(sample[0]) for sample in batch]
        batch_x = self.pad(batch_x)
        batch_y = self.pad(batch_y)
        return batch_x, batch_y,seq_len

    @staticmethod
    def pad(batch, fill=0):
        """
        Pad a batch of samples to maximum length.
        :param batch: list of list
        :param fill: word index to pad, default 0.
        :return: a padded batch
        """
        max_length = max([len(x) for x in batch])
        batch_fill = np.zeros([len(batch),max_length],np.int32)*fill
        for idx, sample in enumerate(batch):
            # if len(sample) < max_length:
            #     batch[idx] = sample + [fill * (max_length - len(sample))]
            batch_fill[idx][:len(sample)]=sample
        return batch_fill


class ToyTrainer(BaseTrainer):
    """
        deprecated
    """

    def __init__(self, train_args):
        super(ToyTrainer, self).__init__(train_args)
        self.test_mode = False
        self.weight = np.random.rand(5, 1)
        self.bias = np.random.rand()
        self._loss = 0
        self._optimizer = None

    def prepare_input(self, data):
        return data[:, :-1], data[:, -1]

    def mode(self, test=False):
        self.model.mode(test)

    def data_forward(self, network, x):
        return np.matmul(x, self.weight) + self.bias

    def grad_backward(self, loss):
        loss.backward()

    def get_loss(self, pred, truth):
        self._loss = np.mean(np.square(pred - truth))
        return self._loss

    def define_optimizer(self):
        self._optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

    def update(self):
        self._optimizer.step()


class WordSegTrainer(BaseTrainer):
    """
        deprecated
    """

    def __init__(self, train_args):
        super(WordSegTrainer, self).__init__(train_args)
        self.id2word = None
        self.word2id = None
        self.id2tag = None
        self.tag2id = None

        self.lstm_batch_size = 8
        self.lstm_seq_len = 32  # Trainer batch_size == lstm_batch_size * lstm_seq_len
        self.hidden_dim = 100
        self.lstm_num_layers = 2
        self.vocab_size = 100
        self.word_emb_dim = 100

        self.hidden = (self.to_var(torch.zeros(2, self.lstm_batch_size, self.word_emb_dim)),
                       self.to_var(torch.zeros(2, self.lstm_batch_size, self.word_emb_dim)))

        self.optimizer = None
        self._loss = None

        self.USE_GPU = False

    def to_var(self, x):
        if torch.cuda.is_available() and self.USE_GPU:
            x = x.cuda()
        return torch.autograd.Variable(x)

    def prepare_input(self, data):
        """
            perform word indices lookup to convert strings into indices
            :param data: list of string, each string contains word + space + [B, M, E, S]
            :return
        """
        word_list = []
        tag_list = []
        for line in data:
            if len(line) > 2:
                tokens = line.split("#")
                word_list.append(tokens[0])
                tag_list.append(tokens[2][0])
        self.id2word = list(set(word_list))
        self.word2id = {word: idx for idx, word in enumerate(self.id2word)}
        self.id2tag = list(set(tag_list))
        self.tag2id = {tag: idx for idx, tag in enumerate(self.id2tag)}
        words = np.array([self.word2id[w] for w in word_list]).reshape(-1, 1)
        tags = np.array([self.tag2id[t] for t in tag_list]).reshape(-1, 1)
        return words, tags

    def mode(self, test=False):
        if test:
            self.model.eval()
        else:
            self.model.train()

    def data_forward(self, network, x):
        """
        :param network: a PyTorch model
        :param x: sequence of length [batch_size], word indices
        :return:
        """
        x = x.reshape(self.lstm_batch_size, self.lstm_seq_len)
        output, self.hidden = network(x, self.hidden)
        return output

    def define_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.85)

    def get_loss(self, predict, truth):
        truth = torch.Tensor(truth)
        self._loss = torch.nn.CrossEntropyLoss(predict, truth)
        return self._loss

    def grad_backward(self, network):
        self.model.zero_grad()
        self._loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 5, norm_type=2)

    def update(self):
        self.optimizer.step()


class POSTrainer(BaseTrainer):
    """
    Trainer for Sequence Modeling

    """
    def __init__(self, train_args):
        super(POSTrainer, self).__init__(train_args)
        self.vocab_size = train_args["vocab_size"]
        self.num_classes = train_args["num_classes"]
        self.max_len = None
        self.mask = None

    def prepare_input(self, data_path):
        """
            To do: Load pkl files of train/dev/test and embedding
        """
        data_train = _pickle.load(open(data_path + "/data_train.pkl", "rb"))
        data_dev = _pickle.load(open(data_path + "/data_train.pkl", "rb"))
        return data_train, data_dev, 0, 1

    def data_forward(self, network, x):
        """
        :param network: the PyTorch model
        :param x: list of list, [batch_size, max_len]
        :return y: [batch_size, num_classes]
        """
        seq_len = [len(seq) for seq in x]
        x = torch.Tensor(x).long()
        self.batch_size = x.size(0)
        self.max_len = x.size(1)
        self.mask = seq_mask(seq_len, self.max_len)
        y = network(x)
        return y

    def mode(self, test=False):
        if test:
            self.model.eval()
        else:
            self.model.train()

    def define_optimizer(self):
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9)

    def grad_backward(self, loss):
        self.model.zero_grad()
        loss.backward()

    def update(self):
        self.optimizer.step()

    def get_loss(self, predict, truth):
        """
        Compute loss given prediction and ground truth.
        :param predict: prediction label vector, [batch_size, num_classes]
        :param truth: ground truth label vector, [batch_size, max_len]
        :return: a scalar
        """
        truth = torch.Tensor(truth)
        if self.loss_func is None:
            if hasattr(self.model, "loss"):
                self.loss_func = self.model.loss
            else:
                self.define_loss()
        loss, prediction = self.loss_func(predict, truth, self.mask, self.batch_size, self.max_len)
        # print("loss={:.2f}".format(loss.data))
        return loss


if __name__ == "__name__":
    train_args = {"epochs": 1, "validate": False, "batch_size": 3, "pickle_path": "./"}
    trainer = BaseTrainer(train_args)
    data_train = [[[1, 2, 3, 4], [0]] * 10] + [[[1, 3, 5, 2], [1]] * 10]
    trainer.batchify(data=data_train)

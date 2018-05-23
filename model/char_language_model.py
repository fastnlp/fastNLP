import os
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from model.base_model import BaseModel


class CharLM(BaseModel):
    """
        Controller of the Character-level Neural Language Model
    """

    def __init__(self):
        super(CharLM, self).__init__()
        """
            Settings
        """
        self.word_embed_dim = 300
        self.char_embedding_dim = 15
        self.cnn_batch_size = 700
        self.lstm_seq_len = 35
        self.lstm_batch_size = 20
        self.vocab_size = 100
        self.num_char = 150

        self.data = None  # named tuple to store all data set
        self.data_ready = False
        self.criterion = nn.CrossEntropyLoss()
        self.loss = None
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, momentum=0.85)
        self.use_gpu = False
        # word_emb_dim == hidden_size / num of hidden units
        self.hidden = (to_var(torch.zeros(2, self.lstm_batch_size, self.word_embed_dim)),
                       to_var(torch.zeros(2, self.lstm_batch_size, self.word_embed_dim)))

        self.model = charLM(self.char_embedding_dim,
                            self.word_embed_dim,
                            self.vocab_size,
                            self.num_char,
                            use_gpu=self.use_gpu)

    def prepare_input(self, raw_text):
        """
            Do some preparation jobs. Transform raw data into input vectors.
        """
        if not self.data_ready:
            # To do: These need to be dropped out from here. (below)
            if os.path.exists("cache/prep.pt") is False:
                self.preprocess()
            objects = torch.load("cache/prep.pt")
            word_dict = objects["word_dict"]
            char_dict = objects["char_dict"]
            max_word_len = objects["max_word_len"]
            self.data_ready = True
            print("word/char dictionary built. Start making inputs.")

            if os.path.exists("cache/data_sets.pt") is False:
                train_text = read_data("./train.txt")
                valid_text = read_data("./valid.txt")
                test_text = read_data("./tests.txt")

                # To do: These need to be dropped out from here. (above)

                input_vec = np.array(text2vec(raw_text, char_dict, max_word_len))

                # Labels are next-word index in word_dict with the same length as inputs
                input_label = np.array([word_dict[w] for w in raw_text[1:]] + [word_dict[raw_text[-1]]])

                category = {"features": input_vec, "label": input_label}
                torch.save(category, "cache/data_sets.pt")
            else:
                data_sets = torch.load("cache/data_sets.pt")
                input_vec = data_sets["features"]
                input_label = data_sets["label"]

            DataTuple = namedtuple("DataTuple", ["feature", "label"])
            self.data = DataTuple(feature=input_vec, label=input_label)

        return self.data.feature, self.data.label

    def mode(self, test=False):
        raise NotImplementedError

    def data_forward(self, x):
        # detach hidden state of LSTM from last batch
        hidden = [state.detach() for state in self.hidden]
        output, self.hidden = self.model(to_var(x), hidden)
        return output

    def grad_backward(self):
        self.model.zero_grad()
        self.loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 5, norm_type=2)
        self.optimizer.step()

    def loss(self, predict, truth):
        self.loss = self.criterion(predict, to_var(truth))
        return self.loss

    @staticmethod
    def preprocess():
        word_dict, char_dict = create_word_char_dict("valid.txt", "train.txt", "tests.txt")
        num_char = len(char_dict)
        char_dict["BOW"] = num_char + 1
        char_dict["EOW"] = num_char + 2
        char_dict["PAD"] = 0
        #  dict of (int, string)
        reverse_word_dict = {value: key for key, value in word_dict.items()}
        max_word_len = max([len(word) for word in word_dict])
        objects = {
            "word_dict": word_dict,
            "char_dict": char_dict,
            "reverse_word_dict": reverse_word_dict,
            "max_word_len": max_word_len
        }
        torch.save(objects, "cache/prep.pt")
        print("Preprocess done.")

    def forward(self, x, hidden):
        lstm_batch_size = x.size()[0]
        lstm_seq_len = x.size()[1]
        x = x.contiguous().view(-1, x.size()[2])
        x = self.char_embed(x)
        x = torch.transpose(x.view(x.size()[0], 1, x.size()[1], -1), 2, 3)
        x = self.conv_layers(x)
        x = self.batch_norm(x)
        x = self.highway1(x)
        x = self.highway2(x)
        x = x.contiguous().view(lstm_batch_size, lstm_seq_len, -1)
        x, hidden = self.lstm(x, hidden)
        x = self.dropout(x)
        x = x.contiguous().view(lstm_batch_size * lstm_seq_len, -1)
        x = self.linear(x)
        return x, hidden


"""
    Global Functions
"""


def batch_generator(x, batch_size):
    # x: [num_words, in_channel, height, width]
    # partitions x into batches
    num_step = x.size()[0] // batch_size
    for t in range(num_step):
        yield x[t * batch_size:(t + 1) * batch_size]


def text2vec(words, char_dict, max_word_len):
    """ Return list of list of int """
    word_vec = []
    for word in words:
        vec = [char_dict[ch] for ch in word]
        if len(vec) < max_word_len:
            vec += [char_dict["PAD"] for _ in range(max_word_len - len(vec))]
        vec = [char_dict["BOW"]] + vec + [char_dict["EOW"]]
        word_vec.append(vec)
    return word_vec


def read_data(file_name):
    with open(file_name, 'r') as f:
        corpus = f.read().lower()
    import re
    corpus = re.sub(r"<unk>", "unk", corpus)
    return corpus.split()


def get_char_dict(vocabulary):
    char_dict = dict()
    count = 1
    for word in vocabulary:
        for ch in word:
            if ch not in char_dict:
                char_dict[ch] = count
                count += 1
    return char_dict


def create_word_char_dict(*file_name):
    text = []
    for file in file_name:
        text += read_data(file)
    word_dict = {word: ix for ix, word in enumerate(set(text))}
    char_dict = get_char_dict(word_dict)
    return word_dict, char_dict


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class Highway(nn.Module):
    """Highway network"""

    def __init__(self, input_size):
        super(Highway, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size, bias=True)
        self.fc2 = nn.Linear(input_size, input_size, bias=True)

    def forward(self, x):
        t = F.sigmoid(self.fc1(x))
        return torch.mul(t, F.relu(self.fc2(x))) + torch.mul(1 - t, x)


class charLM(nn.Module):
    """Character-level Neural Language Model
    CNN + highway network + LSTM
    # Input:
        4D tensor with shape [batch_size, in_channel, height, width]
    # Output:
        2D Tensor with shape [batch_size, vocab_size]
    # Arguments:
        char_emb_dim: the size of each character's embedding
        word_emb_dim: the size of each word's embedding
        vocab_size: num of unique words
        num_char: num of characters
        use_gpu: True or False
    """

    def __init__(self, char_emb_dim, word_emb_dim,
                 vocab_size, num_char, use_gpu):
        super(charLM, self).__init__()
        self.char_emb_dim = char_emb_dim
        self.word_emb_dim = word_emb_dim
        self.vocab_size = vocab_size

        # char embedding layer
        self.char_embed = nn.Embedding(num_char, char_emb_dim)

        # convolutions of filters with different sizes
        self.convolutions = []

        # list of tuples: (the number of filter, width)
        self.filter_num_width = [(25, 1), (50, 2), (75, 3), (100, 4), (125, 5), (150, 6)]

        for out_channel, filter_width in self.filter_num_width:
            self.convolutions.append(
                nn.Conv2d(
                    1,  # in_channel
                    out_channel,  # out_channel
                    kernel_size=(char_emb_dim, filter_width),  # (height, width)
                    bias=True
                )
            )

        self.highway_input_dim = sum([x for x, y in self.filter_num_width])

        self.batch_norm = nn.BatchNorm1d(self.highway_input_dim, affine=False)

        # highway net
        self.highway1 = Highway(self.highway_input_dim)
        self.highway2 = Highway(self.highway_input_dim)

        # LSTM
        self.lstm_num_layers = 2

        self.lstm = nn.LSTM(input_size=self.highway_input_dim,
                            hidden_size=self.word_emb_dim,
                            num_layers=self.lstm_num_layers,
                            bias=True,
                            dropout=0.5,
                            batch_first=True)

        # output layer
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(self.word_emb_dim, self.vocab_size)

        if use_gpu is True:
            for x in range(len(self.convolutions)):
                self.convolutions[x] = self.convolutions[x].cuda()
            self.highway1 = self.highway1.cuda()
            self.highway2 = self.highway2.cuda()
            self.lstm = self.lstm.cuda()
            self.dropout = self.dropout.cuda()
            self.char_embed = self.char_embed.cuda()
            self.linear = self.linear.cuda()
            self.batch_norm = self.batch_norm.cuda()

    def forward(self, x, hidden):
        # Input: Variable of Tensor with shape [num_seq, seq_len, max_word_len+2]
        # Return: Variable of Tensor with shape [num_words, len(word_dict)]
        lstm_batch_size = x.size()[0]
        lstm_seq_len = x.size()[1]

        x = x.contiguous().view(-1, x.size()[2])
        # [num_seq*seq_len, max_word_len+2]

        x = self.char_embed(x)
        # [num_seq*seq_len, max_word_len+2, char_emb_dim]

        x = torch.transpose(x.view(x.size()[0], 1, x.size()[1], -1), 2, 3)
        # [num_seq*seq_len, 1, max_word_len+2, char_emb_dim]

        x = self.conv_layers(x)
        # [num_seq*seq_len, total_num_filters]

        x = self.batch_norm(x)
        # [num_seq*seq_len, total_num_filters]

        x = self.highway1(x)
        x = self.highway2(x)
        # [num_seq*seq_len, total_num_filters]

        x = x.contiguous().view(lstm_batch_size, lstm_seq_len, -1)
        # [num_seq, seq_len, total_num_filters]

        x, hidden = self.lstm(x, hidden)
        # [seq_len, num_seq, hidden_size]

        x = self.dropout(x)
        # [seq_len, num_seq, hidden_size]

        x = x.contiguous().view(lstm_batch_size * lstm_seq_len, -1)
        # [num_seq*seq_len, hidden_size]

        x = self.linear(x)
        # [num_seq*seq_len, vocab_size]
        return x, hidden

    def conv_layers(self, x):
        chosen_list = list()
        for conv in self.convolutions:
            feature_map = F.tanh(conv(x))
            # (batch_size, out_channel, 1, max_word_len-width+1)
            chosen = torch.max(feature_map, 3)[0]
            # (batch_size, out_channel, 1)
            chosen = chosen.squeeze()
            # (batch_size, out_channel)
            chosen_list.append(chosen)

        # (batch_size, total_num_filers)
        return torch.cat(chosen_list, 1)

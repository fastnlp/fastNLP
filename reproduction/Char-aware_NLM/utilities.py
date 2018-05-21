import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F



def batch_generator(x, batch_size):
    # x: [num_words, in_channel, height, width]
    # partitions x into batches
    num_step = x.size()[0] // batch_size
    for t in range(num_step):
        yield x[t*batch_size:(t+1)*batch_size]


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


def seq2vec(input_words, char_embedding, char_embedding_dim, char_table):
    """ convert the input strings into character embeddings """
    # input_words == list of string
    # char_embedding == torch.nn.Embedding
    # char_embedding_dim == int
    # char_table == list of unique chars
    # Returns: tensor of shape [len(input_words), char_embedding_dim, max_word_len+2]
    max_word_len = max([len(word) for word in input_words])
    print("max_word_len={}".format(max_word_len))
    tensor_list = []
    
    start_column = torch.ones(char_embedding_dim, 1)
    end_column = torch.ones(char_embedding_dim, 1)

    for word in input_words:
        # convert string to word embedding
        word_encoding = char_embedding_lookup(word, char_embedding, char_table)
        # add start and end columns
        word_encoding = torch.cat([start_column, word_encoding, end_column], 1)
        # zero-pad right columns
        word_encoding = F.pad(word_encoding, (0, max_word_len-word_encoding.size()[1]+2)).data
        # create dimension
        word_encoding = word_encoding.unsqueeze(0)

        tensor_list.append(word_encoding)

    return torch.cat(tensor_list, 0)


def read_data(file_name):
    # Return: list of strings
    with open(file_name, 'r') as f:
        corpus = f.read().lower()
    import re
    corpus = re.sub(r"<unk>", "unk", corpus)
    return corpus.split()


def get_char_dict(vocabulary):
    # vocabulary == dict of (word, int)
    # Return: dict of (char, int), starting from 1
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
    word_dict = {word:ix for ix, word in enumerate(set(text))}
    char_dict = get_char_dict(word_dict)
    return word_dict, char_dict


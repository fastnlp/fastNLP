
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from model import charLM
from utilities import *
from collections import namedtuple
from test import test


def preprocess():
    
    word_dict, char_dict = create_word_char_dict("valid.txt", "train.txt", "test.txt")
    num_words = len(word_dict)
    num_char  = len(char_dict)
    char_dict["BOW"] = num_char+1
    char_dict["EOW"] = num_char+2
    char_dict["PAD"] = 0
    
    #  dict of (int, string)
    reverse_word_dict = {value:key for key, value in word_dict.items()}
    max_word_len = max([len(word) for word in word_dict])

    objects = {
        "word_dict": word_dict,
        "char_dict": char_dict,
        "reverse_word_dict": reverse_word_dict,
        "max_word_len": max_word_len
    }
    
    torch.save(objects, "cache/prep.pt")
    print("Preprocess done.")


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def train(net, data, opt):
    
    torch.manual_seed(1024)

    train_input = torch.from_numpy(data.train_input)
    train_label = torch.from_numpy(data.train_label)
    valid_input = torch.from_numpy(data.valid_input)
    valid_label = torch.from_numpy(data.valid_label)

    # [num_seq, seq_len, max_word_len+2]
    num_seq = train_input.size()[0] // opt.lstm_seq_len
    train_input = train_input[:num_seq*opt.lstm_seq_len, :]
    train_input = train_input.view(-1, opt.lstm_seq_len, opt.max_word_len+2)

    num_seq = valid_input.size()[0] // opt.lstm_seq_len
    valid_input = valid_input[:num_seq*opt.lstm_seq_len, :]
    valid_input = valid_input.view(-1, opt.lstm_seq_len, opt.max_word_len+2)

    num_epoch = opt.epochs
    num_iter_per_epoch = train_input.size()[0] // opt.lstm_batch_size
    
    learning_rate = opt.init_lr
    old_PPL = 100000
    best_PPL = 100000

    # Log-SoftMax
    criterion = nn.CrossEntropyLoss()
    
    # word_emb_dim == hidden_size / num of hidden units 
    hidden = (to_var(torch.zeros(2, opt.lstm_batch_size, opt.word_embed_dim)), 
              to_var(torch.zeros(2, opt.lstm_batch_size, opt.word_embed_dim)))


    for epoch in range(num_epoch):

        ################  Validation  ####################
        net.eval()
        loss_batch = []
        PPL_batch = []
        iterations = valid_input.size()[0] // opt.lstm_batch_size
        
        valid_generator = batch_generator(valid_input, opt.lstm_batch_size)
        vlabel_generator = batch_generator(valid_label, opt.lstm_batch_size*opt.lstm_seq_len)


        for t in range(iterations):
            batch_input = valid_generator.__next__()
            batch_label = vlabel_generator.__next__()

            hidden = [state.detach() for state in hidden]
            valid_output, hidden = net(to_var(batch_input), hidden)

            length = valid_output.size()[0]

            # [num_sample-1, len(word_dict)] vs [num_sample-1]
            valid_loss = criterion(valid_output, to_var(batch_label))

            PPL = torch.exp(valid_loss.data)

            loss_batch.append(float(valid_loss))
            PPL_batch.append(float(PPL))

        PPL = np.mean(PPL_batch)
        print("[epoch {}] valid PPL={}".format(epoch, PPL))
        print("valid loss={}".format(np.mean(loss_batch)))
        print("PPL decrease={}".format(float(old_PPL - PPL)))

        # Preserve the best model
        if best_PPL > PPL:
            best_PPL = PPL
            torch.save(net.state_dict(), "cache/model.pt")
            torch.save(net, "cache/net.pkl")

        # Adjust the learning rate
        if float(old_PPL - PPL) <= 1.0:
            learning_rate /= 2
            print("halved lr:{}".format(learning_rate))

        old_PPL = PPL

        ##################################################
        #################### Training ####################
        net.train()
        optimizer  = optim.SGD(net.parameters(), 
                               lr = learning_rate, 
                               momentum=0.85)

        # split the first dim
        input_generator = batch_generator(train_input, opt.lstm_batch_size)
        label_generator = batch_generator(train_label, opt.lstm_batch_size*opt.lstm_seq_len)

        for t in range(num_iter_per_epoch):
            batch_input = input_generator.__next__()
            batch_label = label_generator.__next__()

            # detach hidden state of LSTM from last batch
            hidden = [state.detach() for state in hidden]

            output, hidden = net(to_var(batch_input), hidden)
            # [num_word, vocab_size]
            
            loss = criterion(output, to_var(batch_label))

            net.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(net.parameters(), 5, norm_type=2)
            optimizer.step()
            
            
            if (t+1) % 100 == 0:
                print("[epoch {} step {}] train loss={}, Perplexity={}".format(epoch+1, 
                    t+1, float(loss.data), float(np.exp(loss.data))))


    torch.save(net.state_dict(), "cache/model.pt")
    print("Training finished.")


################################################################

if __name__=="__main__":

    word_embed_dim = 300
    char_embedding_dim = 15

    if os.path.exists("cache/prep.pt") is False:
        preprocess()

    objetcs = torch.load("cache/prep.pt")

    word_dict = objetcs["word_dict"]
    char_dict = objetcs["char_dict"]
    reverse_word_dict = objetcs["reverse_word_dict"]
    max_word_len = objetcs["max_word_len"]
    num_words = len(word_dict)

    print("word/char dictionary built. Start making inputs.")


    if os.path.exists("cache/data_sets.pt") is False:
        train_text = read_data("./train.txt")
        valid_text = read_data("./valid.txt")
        test_text  = read_data("./test.txt")

        train_set = np.array(text2vec(train_text, char_dict, max_word_len))
        valid_set = np.array(text2vec(valid_text, char_dict, max_word_len))
        test_set  = np.array(text2vec(test_text,  char_dict, max_word_len))

        # Labels are next-word index in word_dict with the same length as inputs
        train_label = np.array([word_dict[w] for w in train_text[1:]] + [word_dict[train_text[-1]]])
        valid_label = np.array([word_dict[w] for w in valid_text[1:]] + [word_dict[valid_text[-1]]])
        test_label  = np.array([word_dict[w] for w in test_text[1:]] + [word_dict[test_text[-1]]])

        category = {"tdata":train_set, "vdata":valid_set, "test": test_set, 
                    "trlabel":train_label, "vlabel":valid_label, "tlabel":test_label}
        torch.save(category, "cache/data_sets.pt") 
    else:
        data_sets = torch.load("cache/data_sets.pt")
        train_set = data_sets["tdata"]
        valid_set = data_sets["vdata"]
        test_set  = data_sets["test"]
        train_label = data_sets["trlabel"]
        valid_label = data_sets["vlabel"]
        test_label = data_sets["tlabel"]


    DataTuple = namedtuple("DataTuple", 
                "train_input train_label valid_input valid_label test_input test_label")
    data = DataTuple(train_input=train_set,
                     train_label=train_label,
                     valid_input=valid_set,
                     valid_label=valid_label,
                     test_input=test_set,
                     test_label=test_label)

    print("Loaded data sets. Start building network.")



    USE_GPU = True
    cnn_batch_size = 700
    lstm_seq_len = 35
    lstm_batch_size = 20
    # cnn_batch_size == lstm_seq_len * lstm_batch_size

    net = charLM(char_embedding_dim, 
                word_embed_dim, 
                num_words,
                len(char_dict),
                use_gpu=USE_GPU)

    for param in net.parameters():
        nn.init.uniform(param.data, -0.05, 0.05)


    Options = namedtuple("Options", [
            "cnn_batch_size", "init_lr", "lstm_seq_len",
            "max_word_len", "lstm_batch_size", "epochs",
            "word_embed_dim"])
    opt = Options(cnn_batch_size=lstm_seq_len*lstm_batch_size,
                  init_lr=1.0,
                  lstm_seq_len=lstm_seq_len,
                  max_word_len=max_word_len,
                  lstm_batch_size=lstm_batch_size,
                  epochs=35,
                  word_embed_dim=word_embed_dim)


    print("Network built. Start training.")


    # You can stop training anytime by "ctrl+C"
    try:
        train(net, data, opt)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')


    torch.save(net, "cache/net.pkl")
    print("save net")


    test(net, data, opt)

from process import *
from metric import PosMetric
from fastNLP.core.optimizer import Adam, SGD
from model import BiLSTMCRF
from fastNLP import Trainer, Tester
import argparse
import sys
import torch
import torch.nn as nn

## The arg parser to parse the comman line to adjust the network

parser = argparse.ArgumentParser(description='CRF-LSTM Model')
parser.add_argument('--epoch', nargs='?', type=int,
                    help='The epoch times of training', default=300)
parser.add_argument('--rnn_hidden', nargs='?', type=int,
                    help='The hidden dimension of the LSTM', default=200)
parser.add_argument('--word_emb', nargs='?', type=int,
                    help='The embedding size of vocab', default=100)
parser.add_argument('--batch_size', nargs='?', type=int,
                    help='The batch_size of trainer', default=1000)
parser.add_argument('--op', nargs='?', type=int,
                    help='The optimizer for trainer, 0 for Adam, 1 for SGD', default=0)
parser.add_argument('--lr', nargs='?', type=float,
                    help='The learning rate of optimizer', default=0.1)
parser.add_argument('--cuda', nargs='?', type=bool,
                    help='Whether use cuda', default=True)
parser.add_argument('--bilstm', nargs='?', type=bool,
                    help='bilstm or lstm', default=True)
parser.add_argument('--cont', nargs='?', type=str,
                    help='Whether continue from the saved model or from scratch', default="")
parser.add_argument('--mode', nargs='?', type=str, 
                    help="Choose the mode: train&test", default="train")
parser.add_argument('--device', nargs='?', type=int,
                    help="Choose the free device", default=0)
args = parser.parse_args()
torch.cuda.set_device(args.device)

if args.mode != 'train' and args.mode != 'test':
    print ("Please choose the mode train & test")
    sys.exit(0)



def prepare_data():
    ## load the data from the textfile
    datasets = load_data(Conll2003Loader(), [\
                    "./data/conll2003/train.txt",
                    "./data/conll2003/valid.txt",
                    "./data/conll2003/test.txt"
                  ])
    train_data = datasets[0]
    valid_data = datasets[1]
    test_data = datasets[2]
    
    #Lower case the words in the sentences
    lower_case([train_data, valid_data, test_data], "token_list")
    
    ## Build vocab
    vocab = build_vocab([train_data, valid_data, test_data], "token_list")
    speech_vocab = build_vocab([train_data, valid_data, test_data], "label0_list")
    
    ## Build index
    build_index([train_data, valid_data, test_data], "token_list", 'token_index_list', vocab)
    build_index([train_data, valid_data, test_data], "label0_list", 'speech_index_list', speech_vocab)
    
    
    ## Build origin length for each sentence, for mask in the following procedure
    build_origin_len([train_data, valid_data, test_data], "token_list", 'origin_len')
    
    return train_data, valid_data, test_data, vocab, speech_vocab


def workflow():
    
    train_data, valid_data, test_data, vocab, speech_vocab = prepare_data()
    
    
    ## Set the corresponding tags for each dataset, which will be used in the Trainer
    train_data.set_input("token_index_list", "origin_len", "speech_index_list")
    test_data.set_input("token_index_list", "origin_len", "speech_index_list")
    valid_data.set_input("token_index_list", "origin_len", "speech_index_list")

    train_data.set_target("speech_index_list")
    test_data.set_target("speech_index_list")
    valid_data.set_target("speech_index_list")
    
    
    ## Build the model
    config = {
        "vocab_size": len(vocab),
        "word_emb_dim": args.word_emb, 
        "rnn_hidden_units": args.rnn_hidden,
        "num_classes": len(speech_vocab),
        "bi_direction": args.bilstm
    }
    
    ## Load the model from scratch or from saved model
    if args.cont:
        model = torch.load(args.cont)  
    else:
        model = BiLSTMCRF(config)
    
    if args.mode == "train":
        ##Choose the optimizer
        optimizer = Adam(lr=args.lr) if args.op else SGD(lr=args.lr)


        ## Train the model
        trainer = Trainer(
            model=model, 
            train_data=train_data, 
            dev_data=valid_data,
            use_cuda=args.cuda,
            metrics=PosMetric(pred='pred', target='speech_index_list'),
            optimizer=optimizer,
            n_epochs=args.epoch, 
            batch_size=args.batch_size,
            save_path="./save"
        )
        trainer.train()
    
    
    ## Test the model
    tester = Tester(data=test_data, 
                  model=model, 
                  metrics=PosMetric(pred='pred', target='speech_index_list'),
                  use_cuda=args.cuda,
           )
    tester.test()



if __name__ == "__main__":
    workflow()
    

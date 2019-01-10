# BiLSTM-CRF
This project is based on idad of paper https://arxiv.org/pdf/1508.01991.pdf on EMNLP'16, and with the assistance of the fastNLP, (https://github.com/fastnlp/fastNLP), which can facilitate the development of deep learning project based on NLP. 


## Requirement

```pip install fastNLP```

## Usage
```
python main.py -h
usage: main.py [-h]
               [rnn_hidden] [epoch] [word_emb] [batch_size] [op] [lr] [cuda]
               [bilstm] [continue]

CRF-LSTM Model

python main.py -h
usage: main.py [-h] [--epoch [EPOCH]] [--rnn_hidden [RNN_HIDDEN]]
               [--word_emb [WORD_EMB]] [--batch_size [BATCH_SIZE]] [--op [OP]]
               [--lr [LR]] [--cuda [CUDA]] [--bilstm [BILSTM]] [--cont [CONT]]
               [--mode [MODE]] [--device [DEVICE]]

CRF-LSTM Model

optional arguments:
  -h, --help            show this help message and exit
  --epoch [EPOCH]       The epoch times of training
  --rnn_hidden [RNN_HIDDEN]
                        The hidden dimension of the LSTM
  --word_emb [WORD_EMB]
                        The embedding size of vocab
  --batch_size [BATCH_SIZE]
                        The batch_size of trainer
  --op [OP]             The optimizer for trainer, 0 for Adam, 1 for SGD
  --lr [LR]             The learning rate of optimizer
  --cuda [CUDA]         Whether use cuda
  --bilstm [BILSTM]     bilstm or lstm
  --cont [CONT]         Whether continue from the saved model or from scratch
  --mode [MODE]         Choose the mode: train&test
  --device [DEVICE]     Choose the free device
```

## Pretrained Model
The pretrained model is saved at the save/ directory, you can use it by:
```python main.py --cont="save/```

## Jupyter Tutorial
The jupyter file will walk you through the whole process step by step



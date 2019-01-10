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

positional arguments:
  rnn_hidden  The hidden dimension of the LSTM
  epoch       The epoch times of training
  word_emb    The embedding size of vocab
  batch_size  The batch_size of trainer
  op          The optimizer for trainer, 0 for Adam, 1 for SGD
  lr          The learning rate of optimizer
  cuda        Whether use cuda
  bilstm      bilstm or lstm
  continue    Whether continue from the saved model or from scratch

optional arguments:
  -h, --help  show this help message and exit
```

## Pretrained Model
The pretrained model is saved at the save/ directory, you can use it by:
```python main.py --cont="save/```

## Jupyter Tutorial
The jupyter file will walk you through the whole process step by step



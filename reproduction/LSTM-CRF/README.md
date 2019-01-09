# BiLSTM-CRF

## Requirement
(https://github.com/fastnlp/fastNLP)
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
```python main.py --continue="save/```




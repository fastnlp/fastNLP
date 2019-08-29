# Prototype

这是一个很旧版本的reproduction，待修改

## Word2Idx.py
A mapping model between words and indexes

## embedding.py
embedding modules

Contains a simple encapsulation for torch.nn.Embedding

## encoder.py
encoder modules

Contains a simple encapsulation for torch.nn.LSTM

## aggregation.py
aggregation modules

Contains a self-attention model, according to paper "A Structured Self-attentive Sentence Embedding", https://arxiv.org/abs/1703.03130

## predict.py
predict modules

Contains a two layers perceptron for classification

## example.py
An example showing how to use above modules to build a model

Contains a model for sentiment analysis on Yelp dataset, and its training and testing procedures. See https://arxiv.org/abs/1703.03130 for more details.

## prepare.py
A case of using Word2Idx to build Yelp datasets

## dataloader.py
A dataloader for Yelp dataset

It is an iterable object, returning a zero-padded batch every iteration.





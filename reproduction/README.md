## Introduction
This is the implementation of [Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf) paper in PyTorch.
* Dataset is 600k documents extracted from [Yelp 2018](https://www.yelp.com/dataset) customer reviews
* Use [NLTK](http://www.nltk.org/) and [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/) to tokenize documents and sentences
* Both CPU & GPU support
* The best accuracy is 71%, reaching the same performance in the paper

## Requirement
* python 3.6
* pytorch = 0.3.0
* numpy
* gensim
* nltk
* coreNLP

## Parameters
According to the paper and experiment, I set model parameters:
|word embedding dimension|GRU hidden size|GRU layer|word/sentence context vector dimension|
|---|---|---|---|
|200|50|1|100|

And the training parameters:
|Epoch|learning rate|momentum|batch size|
|---|---|---|---|
|3|0.01|0.9|64|

## Run
1. Prepare dataset. Download the [data set](https://www.yelp.com/dataset), and unzip the custom reviews as a file. Use preprocess.py to transform file into data set foe model input.
2. Train the model. The model will trained and autosaved in 'model.dict'
```
python train
```
3. Test the model.
```
python evaluate
```
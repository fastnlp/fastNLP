## Introduction
This is the implementation of [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) paper in PyTorch.
* MRDataset, non-static-model(word2vec rained by Mikolov etal. (2013) on 100 billion words of Google News)
* It can be run in both CPU and GPU
* The best accuracy is 82.61%, which is better than 81.5% in the paper
(by Jingyuan Liu @Fudan University; Email:(fdjingyuan@outlook.com) Welcome to discussion!)

## Requirement
* python 3.6
* pytorch > 0.1
* numpy
* gensim

## Run
STEP 1
install packages like gensim (other needed pakages is the same)
```
pip install gensim
```

STEP 2
install MRdataset and word2vec resources
* MRdataset: you can download the dataset in (https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz)
* word2vec: you can download the file in (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit)

Since this file is more than 1.5G, I did not display in folders. If you download the file, please remember modify the path in Function def word_embeddings(path = './GoogleNews-vectors-negative300.bin/'):


STEP 3
train the model 
```
python train.py
```
you will get the information printed in the screen, like
```
Epoch [1/20], Iter [100/192] Loss: 0.7008
Test Accuracy: 71.869159 %
Epoch [2/20], Iter [100/192] Loss: 0.5957
Test Accuracy: 75.700935 %
Epoch [3/20], Iter [100/192] Loss: 0.4934
Test Accuracy: 78.130841 %

......
Epoch [20/20], Iter [100/192] Loss: 0.0364
Test Accuracy: 81.495327 %
Best Accuracy: 82.616822 %
Best Model: models/cnn.pkl
```

## Hyperparameters
According to the paper and experiment, I set:

|Epoch|Kernel Size|dropout|learning rate|batch size|
|---|---|---|---|---|
|20|\(h,300,100\)|0.5|0.0001|50|

h = [3,4,5]
If the accuracy is not improved, the learning rate will /*0.8.

## Result
I just tried one dataset : MR. (Other 6 dataset in paper SST-1, SST-2, TREC, CR, MPQA)
There are four models in paper: CNN-rand, CNN-static, CNN-non-static, CNN-multichannel.
I have tried CNN-non-static:A model with pre-trained vectors from word2vec. 
All words—including the unknown ones that are randomly initialized and the pretrained vectors are fine-tuned for each task
(which has almost the best performance and the most difficut to implement among the four models)

|Dataset|Class Size|Best Result|Kim's Paper Result|
|---|---|---|---|
|MR|2|82.617%(CNN-non-static)|81.5%(CNN-nonstatic)|



## Reference
* [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
* https://github.com/Shawn1993/cnn-text-classification-pytorch
* https://github.com/junwang4/CNN-sentence-classification-pytorch-2017/blob/master/utils.py


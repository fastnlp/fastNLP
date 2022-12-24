import os

import numpy as np
import pytest

from fastNLP import DataSet, BLEU
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE, _NEED_IMPORT_ONEFLOW, _NEED_IMPORT_JITTOR

if _NEED_IMPORT_PADDLE:
    import paddle

if _NEED_IMPORT_ONEFLOW:
    import oneflow

if _NEED_IMPORT_JITTOR:
    import jittor


@pytest.mark.parametrize('dataset', [
    {'predictions': "the cat is on the mat",'references': "a cat is on the mat"},
    {'predictions': ["the cat is on the mat"],'references': "a cat is on the mat"},
    {'predictions': ["the cat is on the mat"],'references': ["a cat is on the mat"]},
    {'predictions': ["the cat is on the mat"],'references': [["a cat is on the mat"]]},
    {'predictions': "the cat is on the mat",'references': ["a cat is on the mat"]},
    {'predictions': "the cat is on the mat",'references': [["a cat is on the mat"]]},
    {'predictions': "the cat is on the mat", 'references': ["a cat is on the mat", "one cat is on the mat"]},
    {'predictions': ["the cat is on the mat"], 'references': ["a cat is on the mat", "one cat is on the mat"]},
    {'predictions': "the cat is on the mat", 'references': [["a cat is on the mat", "one cat is on the mat"]]},

])
def test_input1(dataset):

   metric = BLEU()
   print(dataset["predictions"])
   metric.update(dataset["predictions"],dataset["references"])
   result = metric.get_metric()
   np.testing.assert_almost_equal(result["bleu"],0.759836)


@pytest.mark.parametrize('dataset', [
    {'predictions': ["the cat is on the mat",'There is a big tree near the park here'],
     'references': ["a cat is on the mat",'A big tree is growing near the park here']},
    {'predictions': ["the cat is on the mat",'There is a big tree near the park here'],
     'references': [["a cat is on the mat"],['A big tree is growing near the park here']]},
])
def test_input2(dataset):

   metric = BLEU()
   print(dataset["predictions"])
   metric.update(dataset["predictions"],dataset["references"])
   result = metric.get_metric()
   np.testing.assert_almost_equal(result["bleu"],0.522605)

@pytest.mark.parametrize('dataset', [
    {'predictions': ["我爱中国,中国爱我"],
     'references': ["我喜欢中国,中国喜欢我"]},
    {'predictions': ["猫坐在垫子上",'公园旁边有棵树'],
     'references': [["猫在那边的垫子"],['一棵树长在公园旁边']]},
])
def test_input3(dataset):

   metric = BLEU()
   metric.update(dataset["predictions"],dataset["references"])
   result = metric.get_metric()
   print(result)

@pytest.mark.paddle
def test_bleu_paddle():
    dataset = DataSet({
        "predictions": ['There is a big tree near the park here',
                        'The sun rises from the northeast with sunshine',
                        'I was late for work today for the rainy',
                        'the cat is on the mat',],
        "references": [['A big tree is growing near the park here'],
                       ["A fierce sun rises in the northeast with sunshine"],
                       ['I went to work too late today for the rainy'],
                       ['a cat is on the mat','one cat is in the mat','cat is in mat','a cat is on a blue mat'],]
    })

    metric = BLEU(backend="paddle")
    for i in range(0, len(dataset), 2):
        prediction1 = dataset[i]["predictions"]
        prediction2 = dataset[i + 1]["predictions"]
        references1 = dataset[i]["references"]
        references2 = dataset[i + 1]["references"]
        metric.update([prediction1, prediction2], [references1, references2])
    my_result = metric.get_metric()
    result = my_result["bleu"]
    np.testing.assert_almost_equal(my_result["bleu"], 0.4181)

    metric = BLEU(backend="paddle", smooth=True)
    for i in range(0, len(dataset), 2):
        prediction1 = dataset[i]["predictions"]
        prediction2 = dataset[i + 1]["predictions"]
        references1 = dataset[i]["references"]
        references2 = dataset[i + 1]["references"]
        metric.update([prediction1, prediction2], [references1, references2])
    my_result = metric.get_metric()
    result = my_result["bleu"]
    np.testing.assert_almost_equal(my_result["bleu"], 0.442571)


@pytest.mark.oneflow
def test_bleu_oneflow():
    dataset = DataSet({
        "predictions": ['There is a big tree near the park here',
                        'The sun rises from the northeast with sunshine',
                        'I was late for work today for the rainy',
                        'the cat is on the mat', ],
        "references": [['A big tree is growing near the park here'],
                       ["A fierce sun rises in the northeast with sunshine"],
                       ['I went to work too late today for the rainy'],
                       ['a cat is on the mat', 'one cat is in the mat', 'cat is in mat', 'a cat is on a blue mat'],]
    })
    metric = BLEU(backend="oneflow")
    for i in range(len(dataset)):
        predictions = dataset[i]["predictions"]
        references = dataset[i]["references"]
        metric.update([predictions], [references])
    my_result = metric.get_metric()
    result = my_result["bleu"]
    np.testing.assert_almost_equal(my_result["bleu"], 0.4181)

    metric = BLEU(backend="oneflow", smooth=True)
    for i in range(len(dataset)):
        predictions = dataset[i]["predictions"]
        references = dataset[i]["references"]
        metric.update([predictions], [references])
    my_result = metric.get_metric()
    result = my_result["bleu"]
    np.testing.assert_almost_equal(my_result["bleu"], 0.442571)


@pytest.mark.jittor
def test_bleu_jittor():
    dataset = DataSet({
        "predictions": ['the cat is on the mat', 'There is a big tree near the park here',
                        'The sun rises from the northeast with sunshine', 'I was late for work today for the rainy'],
        "references": [['a cat is on the mat'], ['A big tree is growing near the park here'],
                       ["A fierce sun rises in the northeast with sunshine"],
                       ['I went to work too late today for the rainy']]
    })
    metric = BLEU(backend="jittor")

    for i in range(len(dataset)):
        predictions = dataset[i]["predictions"]
        references = dataset[i]["references"]
        metric.update([predictions], [references])
    my_result = metric.get_metric()
    result = my_result["bleu"]
    np.testing.assert_almost_equal(my_result["bleu"], 0.4181)

    metric = BLEU(backend="jittor", smooth=True)
    for i in range(len(dataset)):
        predictions = dataset[i]["predictions"]
        references = dataset[i]["references"]
        metric.update([predictions], [references])
    my_result = metric.get_metric()
    result = my_result["bleu"]
    np.testing.assert_almost_equal(my_result["bleu"], 0.442571)
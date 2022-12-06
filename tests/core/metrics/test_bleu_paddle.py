import os

import pytest

from fastNLP import DataSet, Bleu
from fastNLP.envs.imports import _NEED_IMPORT_PADDLE,_NEED_IMPORT_ONEFLOW,_NEED_IMPORT_JITTOR
if _NEED_IMPORT_PADDLE:
    import paddle

if _NEED_IMPORT_ONEFLOW:
    import oneflow

if _NEED_IMPORT_JITTOR:
    import jittor


@pytest.mark.paddle
def test_bleu_paddle():
    dataset = DataSet({
        "predictions": ['the cat is on the mat', 'There is a big tree near the park here',
                        'The sun rises from the northeast with sunshine', 'I was late for work today for the rainy'],
        "references": [['a cat is on the mat'], ['A big tree is growing near the park here'],
                       ["A fierce sun rises in the northeast with sunshine"],
                       ['I went to work too late today for the rainy']]
    })
    metric = Bleu(backend="paddle")
    for i in range(len(dataset)):
        predictions = dataset[i]["predictions"]
        references = dataset[i]["references"]
        metric.update([predictions], [references])
    my_result = metric.get_metric()
    result = my_result["bleu"]
    assert result == 0.4181

    metric = Bleu(backend="paddle",smooth = True)
    for i in range(len(dataset)):
        predictions = dataset[i]["predictions"]
        references = dataset[i]["references"]
        metric.update([predictions], [references])
    my_result = metric.get_metric()
    result = my_result["bleu"]
    assert result == 0.442571


@pytest.mark.oneflow
def test_bleu_oneflow():
    dataset = DataSet({
        "predictions": ['the cat is on the mat', 'There is a big tree near the park here',
                        'The sun rises from the northeast with sunshine', 'I was late for work today for the rainy'],
        "references": [['a cat is on the mat'], ['A big tree is growing near the park here'],
                       ["A fierce sun rises in the northeast with sunshine"],
                       ['I went to work too late today for the rainy']]
    })
    metric = Bleu(backend="oneflow")
    for i in range(len(dataset)):
        predictions = dataset[i]["predictions"]
        references = dataset[i]["references"]
        metric.update([predictions], [references])
    my_result = metric.get_metric()
    result = my_result["bleu"]
    assert result == 0.4181

    metric = Bleu(backend="oneflow",smooth = True)
    for i in range(len(dataset)):
        predictions = dataset[i]["predictions"]
        references = dataset[i]["references"]
        metric.update([predictions], [references])
    my_result = metric.get_metric()
    result = my_result["bleu"]
    assert result == 0.442571


@pytest.mark.oneflow
def test_bleu_jittor():
    dataset = DataSet({
        "predictions": ['the cat is on the mat', 'There is a big tree near the park here',
                        'The sun rises from the northeast with sunshine', 'I was late for work today for the rainy'],
        "references": [['a cat is on the mat'], ['A big tree is growing near the park here'],
                       ["A fierce sun rises in the northeast with sunshine"],
                       ['I went to work too late today for the rainy']]
    })
    metric = Bleu(backend="jittor")
    for i in range(len(dataset)):
        predictions = dataset[i]["predictions"]
        references = dataset[i]["references"]
        metric.update([predictions], [references])
    my_result = metric.get_metric()
    result = my_result["bleu"]
    assert result == 0.4181

    metric = Bleu(backend="jittor",smooth = True)
    for i in range(len(dataset)):
        predictions = dataset[i]["predictions"]
        references = dataset[i]["references"]
        metric.update([predictions], [references])
    my_result = metric.get_metric()
    result = my_result["bleu"]
    assert result == 0.442571




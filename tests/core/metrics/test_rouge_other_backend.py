import numpy as np
import pytest

from fastNLP import DataSet, ROUGE


@pytest.mark.parametrize('use_stemmer', [True, False])
@pytest.mark.paddle
def test_rouge_paddle(use_stemmer):
    dataset = DataSet({
        "predictions": ['the cat is on the mats', 'There is a big trees near the park here',
                        'The sun rises from the northeast with sunshine', 'I was later for work today for the rainy'],
        "references": [['a cat is on the mating'], ['A big tree is growing near the parks here'],
                       ["A fierce sunning rises in the northeast with sunshine"],
                       ['I go to working too later today for the rainy']]
    })
    metric = ROUGE(backend="paddle", use_stemmer=use_stemmer)
    for i in range(0, len(dataset), 2):
        prediction1 = dataset[i]["predictions"]
        prediction2 = dataset[i + 1]["predictions"]
        references1 = dataset[i]["references"]
        references2 = dataset[i + 1]["references"]
        metric.update([prediction1, prediction2], [references1, references2])
    results = metric.get_metric()
    if use_stemmer:
        np.testing.assert_almost_equal(results['rouge1_fmeasure'], 0.7495700120925903)
        np.testing.assert_almost_equal(results['rouge2_fmeasure'], 0.5278186202049255)
        np.testing.assert_almost_equal(results['rougeL_fmeasure'], 0.6954764127731323)
    else:
        np.testing.assert_almost_equal(results['rouge1_fmeasure'], 0.6382868885993958)
        np.testing.assert_almost_equal(results['rouge2_fmeasure'], 0.4007352888584137)
        np.testing.assert_almost_equal(results['rougeL_fmeasure'], 0.6105090975761414)



@pytest.mark.parametrize('use_stemmer', [True, False])
@pytest.mark.oneflow
def test_rouge_oneflow(use_stemmer):
    dataset = DataSet({
        "predictions": ['the cat is on the mats', 'There is a big trees near the park here',
                        'The sun rises from the northeast with sunshine', 'I was later for work today for the rainy'],
        "references": [['a cat is on the mating'], ['A big tree is growing near the parks here'],
                       ["A fierce sunning rises in the northeast with sunshine"],
                       ['I go to working too later today for the rainy']]
    })
    metric = ROUGE(backend="oneflow",use_stemmer=use_stemmer)
    for i in range(len(dataset)):
        predictions = dataset[i]["predictions"]
        references = dataset[i]["references"]
        metric.update([predictions], [references])
    results = metric.get_metric()
    print(results)
    if use_stemmer:
        np.testing.assert_almost_equal(results['rouge1_fmeasure'], 0.7495700120925903)
        np.testing.assert_almost_equal(results['rouge2_fmeasure'], 0.5278186202049255)
        np.testing.assert_almost_equal(results['rougeL_fmeasure'], 0.6954764127731323)
    else:
        np.testing.assert_almost_equal(results['rouge1_fmeasure'], 0.6382868885993958)
        np.testing.assert_almost_equal(results['rouge2_fmeasure'], 0.4007352888584137)
        np.testing.assert_almost_equal(results['rougeL_fmeasure'], 0.6105090975761414)


@pytest.mark.parametrize('use_stemmer', [True, False])
@pytest.mark.jittor
def test_rouge_jittor(use_stemmer):
    dataset = DataSet({
        "predictions": ['the cat is on the mats', 'There is a big trees near the park here',
                        'The sun rises from the northeast with sunshine', 'I was later for work today for the rainy'],
        "references": [['a cat is on the mating'], ['A big tree is growing near the parks here'],
                       ["A fierce sunning rises in the northeast with sunshine"],
                       ['I go to working too later today for the rainy']]
    })
    metric = ROUGE(backend="jittor",use_stemmer=use_stemmer)
    for i in range(len(dataset)):
        predictions = dataset[i]["predictions"]
        references = dataset[i]["references"]
        metric.update([predictions], [references])
    results = metric.get_metric()
    if use_stemmer:
        np.testing.assert_almost_equal(results['rouge1_fmeasure'], 0.7495700120925903)
        np.testing.assert_almost_equal(results['rouge2_fmeasure'], 0.5278186202049255)
        np.testing.assert_almost_equal(results['rougeL_fmeasure'], 0.6954764127731323)
    else:
        np.testing.assert_almost_equal(results['rouge1_fmeasure'], 0.6382868885993958)
        np.testing.assert_almost_equal(results['rouge2_fmeasure'], 0.4007352888584137)
        np.testing.assert_almost_equal(results['rougeL_fmeasure'], 0.6105090975761414)

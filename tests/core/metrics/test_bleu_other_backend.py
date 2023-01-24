import numpy as np
import pytest

from fastNLP import BLEU, DataSet


@pytest.mark.paddle
def test_bleu_paddle():
    dataset = DataSet({
        'predictions': [
            'There is a big tree near the park here',
            'The sun rises from the northeast with sunshine',
            'I was late for work today for the rainy',
            'the cat is on the mat',
        ],
        'references': [
            ['A big tree is growing near the park here'],
            ['A fierce sun rises in the northeast with sunshine'],
            ['I went to work too late today for the rainy'],
            [
                'a cat is on the mat', 'one cat is in the mat',
                'cat is in mat', 'a cat is on a blue mat'
            ],
        ]
    })

    metric = BLEU(backend='paddle')
    for i in range(0, len(dataset), 2):
        prediction1 = dataset[i]['predictions']
        prediction2 = dataset[i + 1]['predictions']
        references1 = dataset[i]['references']
        references2 = dataset[i + 1]['references']
        metric.update([prediction1, prediction2], [references1, references2])
    my_result = metric.get_metric()
    np.testing.assert_almost_equal(my_result['bleu'], 0.4181)

    metric = BLEU(backend='paddle', smooth=True)
    for i in range(0, len(dataset), 2):
        prediction1 = dataset[i]['predictions']
        prediction2 = dataset[i + 1]['predictions']
        references1 = dataset[i]['references']
        references2 = dataset[i + 1]['references']
        metric.update([prediction1, prediction2], [references1, references2])
    my_result = metric.get_metric()
    np.testing.assert_almost_equal(my_result['bleu'], 0.442571)


@pytest.mark.oneflow
def test_bleu_oneflow():
    dataset = DataSet({
        'predictions': [
            'There is a big tree near the park here',
            'The sun rises from the northeast with sunshine',
            'I was late for work today for the rainy',
            'the cat is on the mat',
        ],
        'references': [
            ['A big tree is growing near the park here'],
            ['A fierce sun rises in the northeast with sunshine'],
            ['I went to work too late today for the rainy'],
            [
                'a cat is on the mat', 'one cat is in the mat',
                'cat is in mat', 'a cat is on a blue mat'
            ],
        ]
    })
    metric = BLEU(backend='oneflow')
    for i in range(len(dataset)):
        predictions = dataset[i]['predictions']
        references = dataset[i]['references']
        metric.update([predictions], [references])
    my_result = metric.get_metric()
    np.testing.assert_almost_equal(my_result['bleu'], 0.4181)

    metric = BLEU(backend='oneflow', smooth=True)
    for i in range(len(dataset)):
        predictions = dataset[i]['predictions']
        references = dataset[i]['references']
        metric.update([predictions], [references])
    my_result = metric.get_metric()
    np.testing.assert_almost_equal(my_result['bleu'], 0.442571)


@pytest.mark.jittor
def test_bleu_jittor():
    dataset = DataSet({
        'predictions': [
            'the cat is on the mat', 'There is a big tree near the park here',
            'The sun rises from the northeast with sunshine',
            'I was late for work today for the rainy'
        ],
        'references': [['a cat is on the mat'],
                       ['A big tree is growing near the park here'],
                       ['A fierce sun rises in the northeast with sunshine'],
                       ['I went to work too late today for the rainy']]
    })
    metric = BLEU(backend='jittor')

    for i in range(len(dataset)):
        predictions = dataset[i]['predictions']
        references = dataset[i]['references']
        metric.update([predictions], [references])
    my_result = metric.get_metric()
    np.testing.assert_almost_equal(my_result['bleu'], 0.4181)

    metric = BLEU(backend='jittor', smooth=True)
    for i in range(len(dataset)):
        predictions = dataset[i]['predictions']
        references = dataset[i]['references']
        metric.update([predictions], [references])
    my_result = metric.get_metric()
    np.testing.assert_almost_equal(my_result['bleu'], 0.442571)

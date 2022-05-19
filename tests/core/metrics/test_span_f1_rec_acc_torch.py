import pytest

from collections import Counter
import os, sys
import copy
from functools import partial

import numpy as np
import socket

# from multiprocessing import Pool, set_start_method
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.core.metrics import SpanFPreRecMetric
from fastNLP.core.dataset import DataSet
from fastNLP.envs.imports import _NEED_IMPORT_TORCH
from .utils import find_free_network_port, setup_ddp
if _NEED_IMPORT_TORCH:
    import torch
    import torch.distributed
    from torch.multiprocessing import Pool, set_start_method
else:
    from fastNLP.core.utils.dummy_class import DummyClass as set_start_method

set_start_method("spawn", force=True)


def _generate_tags(encoding_type, number_labels=4):
    """

    :param encoding_type: 例如BIOES, BMES, BIO等
    :param number_labels: 多少个label，大于1
    :return:
    """
    vocab = {}
    for i in range(number_labels):
        label = str(i)
        for tag in encoding_type:
            if tag == 'O':
                if tag not in vocab:
                    vocab['O'] = len(vocab) + 1
                continue
            vocab['{}-{}'.format(tag, label)] = len(vocab) + 1  # 其实表达的是这个的count
    return vocab


NUM_PROCESSES = 2
pool = None


def _test(local_rank: int,
          world_size: int,
          device: "torch.device",
          dataset: DataSet,
          metric_class,
          metric_kwargs,
          sklearn_metric) -> None:
    # metric 应该是每个进程有自己的一个 instance，所以在 _test 里面实例化
    metric = metric_class(**metric_kwargs)
    # dataset 也类似（每个进程有自己的一个）
    dataset = copy.deepcopy(dataset)
    metric.to(device)
    # 把数据拆到每个 GPU 上，有点模仿 DistributedSampler 的感觉，但这里数据单位是一个 batch（即每个 i 取了一个 batch 到自己的 GPU 上）
    for i in range(local_rank, len(dataset), world_size):
        pred, tg, seq_len = dataset[i]['pred'].to(device), dataset[i]['tg'].to(device), dataset[i]['seq_len']
        print(tg, seq_len)
        metric.update(pred, tg, seq_len)

    my_result = metric.get_metric()
    print(my_result)
    print(sklearn_metric)
    assert my_result == sklearn_metric


@pytest.mark.torch
class TestSpanFPreRecMetric:

    def test_case1(self):
        from fastNLP.core.metrics.span_f1_pre_rec_metric import _bmes_tag_to_spans
        from fastNLP.core.metrics.span_f1_pre_rec_metric import _bio_tag_to_spans

        bmes_lst = ['M-8', 'S-2', 'S-0', 'B-9', 'B-6', 'E-5', 'B-7', 'S-2', 'E-7', 'S-8']
        bio_lst = ['O-8', 'O-2', 'B-0', 'O-9', 'I-6', 'I-5', 'I-7', 'I-2', 'I-7', 'O-8']
        expect_bmes_res = set()
        expect_bmes_res.update([('8', (0, 1)), ('2', (1, 2)), ('0', (2, 3)), ('9', (3, 4)), ('6', (4, 5)),
                                ('5', (5, 6)), ('7', (6, 7)), ('2', (7, 8)), ('7', (8, 9)), ('8', (9, 10))])
        expect_bio_res = set()
        expect_bio_res.update([('7', (8, 9)), ('0', (2, 3)), ('2', (7, 8)), ('5', (5, 6)),
                               ('6', (4, 5)), ('7', (6, 7))])
        assert expect_bmes_res == set(_bmes_tag_to_spans(bmes_lst))
        assert expect_bio_res == set(_bio_tag_to_spans(bio_lst))

    def test_case2(self):
        # 测试不带label的
        from fastNLP.core.metrics.span_f1_pre_rec_metric import _bmes_tag_to_spans
        from fastNLP.core.metrics.span_f1_pre_rec_metric import _bio_tag_to_spans

        bmes_lst = ['B', 'E', 'B', 'S', 'B', 'M', 'E', 'M', 'B', 'E']
        bio_lst = ['I', 'B', 'O', 'O', 'I', 'O', 'I', 'B', 'O', 'O']
        expect_bmes_res = set()
        expect_bmes_res.update([('', (0, 2)), ('', (2, 3)), ('', (3, 4)), ('', (4, 7)), ('', (7, 8)), ('', (8, 10))])
        expect_bio_res = set()
        expect_bio_res.update([('', (7, 8)), ('', (6, 7)), ('', (4, 5)), ('', (0, 1)), ('', (1, 2))])
        assert expect_bmes_res == set(_bmes_tag_to_spans(bmes_lst))
        assert expect_bio_res == set(_bio_tag_to_spans(bio_lst))

    def test_case3(self):
        number_labels = 4
        # bio tag
        fastnlp_bio_vocab = Vocabulary(unknown=None, padding=None)
        fastnlp_bio_vocab.word_count = Counter(_generate_tags('BIO', number_labels))
        fastnlp_bio_metric = SpanFPreRecMetric(tag_vocab=fastnlp_bio_vocab, only_gross=False,
                                               aggregate_when_get_metric=True)
        bio_sequence = torch.FloatTensor([[[-0.4424, -0.4579, -0.7376,  1.8129,  0.1316,  1.6566, -1.2169,
                                            -0.3782,  0.8240],
                                           [-1.2348, -0.1876, -0.1462, -0.4834, -0.6692, -0.9735,  1.1563,
                                            -0.3562, -1.4116],
                                           [ 1.6550, -0.9555,  0.3782, -1.3160, -1.5835, -0.3443, -1.7858,
                                             2.0023,  0.7075],
                                           [-0.3772, -0.5447, -1.5631,  1.1614,  1.4598, -1.2764,  0.5186,
                                            0.3832, -0.1540],
                                           [-0.1011,  0.0600,  1.1090, -0.3545,  0.1284,  1.1484, -1.0120,
                                            -1.3508, -0.9513],
                                           [ 1.8948,  0.8627, -2.1359,  1.3740, -0.7499,  1.5019,  0.6919,
                                             -0.0842, -0.4294]],

                                          [[-0.2802,  0.6941, -0.4788, -0.3845,  1.7752,  1.2950, -1.9490,
                                            -1.4138, -0.8853],
                                           [-1.3752, -0.5457, -0.5305,  0.4018,  0.2934,  0.7931,  2.3845,
                                            -1.0726,  0.0364],
                                           [ 0.3621,  0.2609,  0.1269, -0.5950,  0.7212,  0.5959,  1.6264,
                                             -0.8836, -0.9320],
                                           [ 0.2003, -1.0758, -1.1560, -0.6472, -1.7549,  0.1264,  0.6044,
                                             -1.6857,  1.1571],
                                           [ 1.4277, -0.4915,  0.4496,  2.2027,  0.0730, -3.1792, -0.5125,
                                             -0.5837,  1.0184],
                                           [ 1.9495,  1.7145, -0.2143, -0.1230, -0.2205,  0.8250,  0.4943,
                                             -0.9025,  0.0864]]])
        bio_target = torch.LongTensor([[3, 6, 0, 8, 2, 4], [4, 1, 7, 0, 4, 7]])
        fastnlp_bio_metric.update(bio_sequence, bio_target, [6, 6])
        expect_bio_res = {'pre-1': 0.333333, 'rec-1': 0.333333, 'f-1': 0.333333, 'pre-2': 0.5, 'rec-2': 0.5,
                          'f-2': 0.5, 'pre-0': 0.0, 'rec-0': 0.0, 'f-0': 0.0, 'pre-3': 0.0, 'rec-3': 0.0,
                          'f-3': 0.0, 'pre': 0.222222, 'rec': 0.181818, 'f': 0.2}
        assert expect_bio_res == fastnlp_bio_metric.get_metric()
        # print(fastnlp_bio_metric.get_metric())

    def test_case4(self):
        # bmes tag
        def _generate_samples():
            target = []
            seq_len = []
            vocab = Vocabulary(unknown=None, padding=None)
            for i in range(3):
                target_i = []
                seq_len_i = 0
                for j in range(1, 10):
                    word_len = np.random.randint(1, 5)
                    seq_len_i += word_len
                    if word_len == 1:
                        target_i.append('S')
                    else:
                        target_i.append('B')
                        target_i.extend(['M'] * (word_len - 2))
                        target_i.append('E')
                vocab.add_word_lst(target_i)
                target.append(target_i)
                seq_len.append(seq_len_i)
            target_ = np.zeros((3, max(seq_len)))
            for i in range(3):
                target_i = [vocab.to_index(t) for t in target[i]]
                target_[i, :seq_len[i]] = target_i
            return target_, target, seq_len, vocab

        def get_eval(raw_target, pred, vocab, seq_len):
            pred = pred.argmax(dim=-1).tolist()
            tp = 0
            gold = 0
            seg = 0
            pred_target = []
            for i in range(len(seq_len)):
                tags = [vocab.to_word(p) for p in pred[i][:seq_len[i]]]
                spans = []
                prev_bmes_tag = None
                for idx, tag in enumerate(tags):
                    if tag in ('B', 'S'):
                        spans.append([idx, idx])
                    elif tag in ('M', 'E') and prev_bmes_tag in ('B', 'M'):
                        spans[-1][1] = idx
                    else:
                        spans.append([idx, idx])
                    prev_bmes_tag = tag
                tmp = []
                for span in spans:
                    if span[1] - span[0] > 0:
                        tmp.extend(['B'] + ['M'] * (span[1] - span[0] - 1) + ['E'])
                    else:
                        tmp.append('S')
                pred_target.append(tmp)
            for i in range(len(seq_len)):
                raw_pred = pred_target[i]
                start = 0
                for j in range(seq_len[i]):
                    if raw_target[i][j] in ('E', 'S'):
                        flag = True
                        for k in range(start, j + 1):
                            if raw_target[i][k] != raw_pred[k]:
                                flag = False
                                break
                        if flag:
                            tp += 1
                        start = j + 1
                        gold += 1
                    if raw_pred[j] in ('E', 'S'):
                        seg += 1

            pre = round(tp / seg, 6)
            rec = round(tp / gold, 6)
            return {'f': round(2 * pre * rec / (pre + rec), 6), 'pre': pre, 'rec': rec}

        target, raw_target, seq_len, vocab = _generate_samples()
        pred = torch.randn(3, max(seq_len), 4)

        expected_metric = get_eval(raw_target, pred, vocab, seq_len)
        metric = SpanFPreRecMetric(tag_vocab=vocab, encoding_type='bmes')
        metric.update(pred, torch.from_numpy(target), seq_len)
        # print(metric.get_metric(reset=False))
        # print(expected_metric)
        metric_value = metric.get_metric()
        for key, value in expected_metric.items():
            assert np.allclose(value, metric_value[key])

    def test_auto_encoding_type_infer(self):
        #  检查是否可以自动check encode的类型
        vocabs = {}
        import random
        for encoding_type in ['bio', 'bioes', 'bmeso']:
            vocab = Vocabulary(unknown=None, padding=None)
            for i in range(random.randint(10, 100)):
                label = str(random.randint(1, 10))
                for tag in encoding_type:
                    if tag != 'o':
                        vocab.add_word(f'{tag}-{label}')
                    else:
                        vocab.add_word('o')
            vocabs[encoding_type] = vocab
        for e in ['bio', 'bioes', 'bmeso']:
            metric = SpanFPreRecMetric(tag_vocab=vocabs[e])
            assert metric.encoding_type == e

        bmes_vocab = _generate_tags('bmes')
        vocab = Vocabulary()
        for tag, index in bmes_vocab.items():
            vocab.add_word(tag)
        metric = SpanFPreRecMetric(tag_vocab=vocab)
        assert metric.encoding_type == 'bmes'

        # 一些无法check的情况
        vocab = Vocabulary()
        for i in range(10):
            vocab.add_word(str(i))
        with pytest.raises(Exception):
            metric = SpanFPreRecMetric(vocab)

    def test_encoding_type(self):
        # 检查传入的tag_vocab与encoding_type不符合时，是否会报错
        vocabs = {}
        import random
        from itertools import product
        for encoding_type in ['bio', 'bioes', 'bmeso']:
            vocab = Vocabulary(unknown=None, padding=None)
            for i in range(random.randint(10, 100)):
                label = str(random.randint(1, 10))
                for tag in encoding_type:
                    if tag != 'o':
                        vocab.add_word(f'{tag}-{label}')
                    else:
                        vocab.add_word('o')
            vocabs[encoding_type] = vocab
        for e1, e2 in product(['bio', 'bioes', 'bmeso'], ['bio', 'bioes', 'bmeso']):
            if e1 == e2:
                metric = SpanFPreRecMetric(tag_vocab=vocabs[e1], encoding_type=e2)
            else:
                s2 = set(e2)
                s2.update(set(e1))
                if s2 == set(e2):
                    continue
                with pytest.raises(AssertionError):
                    metric = SpanFPreRecMetric(tag_vocab=vocabs[e1], encoding_type=e2)
        for encoding_type in ['bio', 'bioes', 'bmeso']:
            with pytest.raises(AssertionError):
                metric = SpanFPreRecMetric(tag_vocab=vocabs[encoding_type], encoding_type='bmes')

        with pytest.warns(Warning):
            vocab = Vocabulary(unknown=None, padding=None).add_word_lst(list('bmes'))
            metric = SpanFPreRecMetric(tag_vocab=vocab, encoding_type='bmeso')
            vocab = Vocabulary().add_word_lst(list('bmes'))
            metric = SpanFPreRecMetric(tag_vocab=vocab, encoding_type='bmeso')

    def test_case5(self):
        # global pool
        pool = Pool(NUM_PROCESSES)
        master_port = find_free_network_port()
        pool.starmap(setup_ddp, [(rank, NUM_PROCESSES, master_port) for rank in range(NUM_PROCESSES)])
        number_labels = 4
        # bio tag
        fastnlp_bio_vocab = Vocabulary(unknown=None, padding=None)
        fastnlp_bio_vocab.word_count = Counter(_generate_tags('BIO', number_labels))
        # fastnlp_bio_metric = SpanFPreRecMetric(tag_vocab=fastnlp_bio_vocab, only_gross=False)
        dataset = DataSet({'pred': [
            torch.FloatTensor([[[-0.4424, -0.4579, -0.7376, 1.8129, 0.1316, 1.6566, -1.2169,
                                 -0.3782, 0.8240],
                                [-1.2348, -0.1876, -0.1462, -0.4834, -0.6692, -0.9735, 1.1563,
                                 -0.3562, -1.4116],
                                [1.6550, -0.9555, 0.3782, -1.3160, -1.5835, -0.3443, -1.7858,
                                 2.0023, 0.7075],
                                [-0.3772, -0.5447, -1.5631, 1.1614, 1.4598, -1.2764, 0.5186,
                                 0.3832, -0.1540],
                                [-0.1011, 0.0600, 1.1090, -0.3545, 0.1284, 1.1484, -1.0120,
                                 -1.3508, -0.9513],
                                [1.8948, 0.8627, -2.1359, 1.3740, -0.7499, 1.5019, 0.6919,
                                 -0.0842, -0.4294]]

                               ]),
            torch.FloatTensor([
                [[-0.2802, 0.6941, -0.4788, -0.3845, 1.7752, 1.2950, -1.9490,
                  -1.4138, -0.8853],
                 [-1.3752, -0.5457, -0.5305, 0.4018, 0.2934, 0.7931, 2.3845,
                  -1.0726, 0.0364],
                 [0.3621, 0.2609, 0.1269, -0.5950, 0.7212, 0.5959, 1.6264,
                  -0.8836, -0.9320],
                 [0.2003, -1.0758, -1.1560, -0.6472, -1.7549, 0.1264, 0.6044,
                  -1.6857, 1.1571],
                 [1.4277, -0.4915, 0.4496, 2.2027, 0.0730, -3.1792, -0.5125,
                  -0.5837, 1.0184],
                 [1.9495, 1.7145, -0.2143, -0.1230, -0.2205, 0.8250, 0.4943,
                  -0.9025, 0.0864]]
            ])
                                   ],
                           'tg': [
            torch.LongTensor([[3, 6, 0, 8, 2, 4]]),
            torch.LongTensor([[4, 1, 7, 0, 4, 7]])
                                 ],
                           'seq_len': [
            [6], [6]
                                      ]})
        metric_kwargs = {
            'tag_vocab': fastnlp_bio_vocab,
            'only_gross': False,
            'aggregate_when_get_metric': True
        }
        expect_bio_res = {'pre-1': 0.333333, 'rec-1': 0.333333, 'f-1': 0.333333, 'pre-2': 0.5, 'rec-2': 0.5,
                          'f-2': 0.5, 'pre-0': 0.0, 'rec-0': 0.0, 'f-0': 0.0, 'pre-3': 0.0, 'rec-3': 0.0,
                          'f-3': 0.0, 'pre': 0.222222, 'rec': 0.181818, 'f': 0.2}
        processes = NUM_PROCESSES

        pool.starmap(
            partial(
                _test,
                dataset=dataset,
                metric_class=SpanFPreRecMetric,
                metric_kwargs=metric_kwargs,
                sklearn_metric=expect_bio_res,
            ),
            [(rank, processes, torch.device(f'cuda:{rank}')) for rank in range(processes)]
        )
        pool.close()
        pool.join()

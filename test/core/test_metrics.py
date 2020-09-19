import unittest
from collections import Counter

import numpy as np
import torch
from fastNLP import AccuracyMetric
from fastNLP.core.metrics import (ClassifyFPreRecMetric, CMRC2018Metric,
                                  ConfusionMatrixMetric, SpanFPreRecMetric,
                                  _accuracy_topk, _pred_topk)
from fastNLP.core.vocabulary import Vocabulary


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


def _convert_res_to_fastnlp_res(metric_result):
    allen_result = {}
    key_map = {'f1-measure-overall': "f", "recall-overall": "rec", "precision-overall": "pre"}
    for key, value in metric_result.items():
        if key in key_map:
            key = key_map[key]
        else:
            label = key.split('-')[-1]
            if key.startswith('f1'):
                key = 'f-{}'.format(label)
            else:
                key = '{}-{}'.format(key[:3], label)
        allen_result[key] = round(value, 6)
    return allen_result



class TestConfusionMatrixMetric(unittest.TestCase):
    def test_ConfusionMatrixMetric1(self):
        pred_dict = {"pred": torch.zeros(4,3)}
        target_dict = {'target': torch.zeros(4)}
        metric = ConfusionMatrixMetric()

        metric(pred_dict=pred_dict, target_dict=target_dict)
        print(metric.get_metric())

    def test_ConfusionMatrixMetric2(self):
        # (2) with corrupted size

        with self.assertRaises(Exception):
            pred_dict = {"pred": torch.zeros(4, 3, 2)}
            target_dict = {'target': torch.zeros(4)}
            metric = ConfusionMatrixMetric()

            metric(pred_dict=pred_dict, target_dict=target_dict, )
            print(metric.get_metric())

    def test_ConfusionMatrixMetric3(self):
    # (3) the second batch is corrupted size
        with self.assertRaises(Exception):
            metric = ConfusionMatrixMetric()
            pred_dict = {"pred": torch.zeros(4, 3, 2)}
            target_dict = {'target': torch.zeros(4, 3)}
            metric(pred_dict=pred_dict, target_dict=target_dict)
            
            pred_dict = {"pred": torch.zeros(4, 3, 2)}
            target_dict = {'target': torch.zeros(4)}
            metric(pred_dict=pred_dict, target_dict=target_dict)
            
            print(metric.get_metric())

    def test_ConfusionMatrixMetric4(self):
    # (4) check reset
        metric = ConfusionMatrixMetric()
        pred_dict = {"pred": torch.randn(4, 3, 2)}
        target_dict = {'target': torch.ones(4, 3)}
        metric(pred_dict=pred_dict, target_dict=target_dict)
        res = metric.get_metric()
        self.assertTrue(isinstance(res, dict))
        print(res)

    def test_ConfusionMatrixMetric5(self):
    # (5) check numpy array is not acceptable

        with self.assertRaises(Exception):
            metric = ConfusionMatrixMetric()
            pred_dict = {"pred": np.zeros((4, 3, 2))}
            target_dict = {'target': np.zeros((4, 3))}
            metric(pred_dict=pred_dict, target_dict=target_dict)

    def test_ConfusionMatrixMetric6(self):
    # (6) check map, match
        metric = ConfusionMatrixMetric(pred='predictions', target='targets')
        pred_dict = {"predictions": torch.randn(4, 3, 2)}
        target_dict = {'targets': torch.zeros(4, 3)}
        metric(pred_dict=pred_dict, target_dict=target_dict)
        res = metric.get_metric()
        print(res)

    def test_ConfusionMatrixMetric7(self):
        # (7) check map, include unused
        metric = ConfusionMatrixMetric(pred='prediction', target='targets')
        pred_dict = {"prediction": torch.zeros(4, 3, 2), 'unused': 1}
        target_dict = {'targets': torch.zeros(4, 3)}
        metric(pred_dict=pred_dict, target_dict=target_dict)

    def test_ConfusionMatrixMetric8(self):
        # (8) check _fast_metric
        with self.assertRaises(Exception):
            metric = ConfusionMatrixMetric()
            pred_dict = {"predictions": torch.zeros(4, 3, 2), "seq_len": torch.ones(3) * 3}
            target_dict = {'targets': torch.zeros(4, 3)}
            metric(pred_dict=pred_dict, target_dict=target_dict)
            print(metric.get_metric())


    def test_duplicate(self):
        # 0.4.1的潜在bug，不能出现形参重复的情况
        metric = ConfusionMatrixMetric(pred='predictions', target='targets')
        pred_dict = {"predictions": torch.zeros(4, 3, 2), "seq_len": torch.ones(4) * 3, 'pred':0}
        target_dict = {'targets':torch.zeros(4, 3), 'target': 0}
        metric(pred_dict=pred_dict, target_dict=target_dict)
        print(metric.get_metric())


    def test_seq_len(self):
        N = 256
        seq_len = torch.zeros(N).long()
        seq_len[0] = 2
        pred = {'pred': torch.ones(N, 2)}
        target = {'target': torch.ones(N, 2), 'seq_len': seq_len}
        metric = ConfusionMatrixMetric()
        metric(pred_dict=pred, target_dict=target)
        metric.get_metric(reset=False)
        seq_len[1:] = 1
        metric(pred_dict=pred, target_dict=target)
        metric.get_metric()

    def test_vocab(self):
        vocab = Vocabulary()
        word_list = "this is a word list".split()
        vocab.update(word_list)
        
        pred_dict = {"pred": torch.zeros(4,3)}
        target_dict = {'target': torch.zeros(4)}
        metric = ConfusionMatrixMetric(vocab=vocab)
        metric(pred_dict=pred_dict, target_dict=target_dict)
        print(metric.get_metric())



class TestAccuracyMetric(unittest.TestCase):
    def test_AccuracyMetric1(self):
        # (1) only input, targets passed
        pred_dict = {"pred": torch.zeros(4, 3)}
        target_dict = {'target': torch.zeros(4)}
        metric = AccuracyMetric()
        
        metric(pred_dict=pred_dict, target_dict=target_dict)
        print(metric.get_metric())
    
    def test_AccuracyMetric2(self):
        # (2) with corrupted size
        try:
            pred_dict = {"pred": torch.zeros(4, 3, 2)}
            target_dict = {'target': torch.zeros(4)}
            metric = AccuracyMetric()
            
            metric(pred_dict=pred_dict, target_dict=target_dict, )
            print(metric.get_metric())
        except Exception as e:
            print(e)
            return
        print("No exception catches.")
    
    def test_AccuracyMetric3(self):
        # (3) the second batch is corrupted size
        try:
            metric = AccuracyMetric()
            pred_dict = {"pred": torch.zeros(4, 3, 2)}
            target_dict = {'target': torch.zeros(4, 3)}
            metric(pred_dict=pred_dict, target_dict=target_dict)
            
            pred_dict = {"pred": torch.zeros(4, 3, 2)}
            target_dict = {'target': torch.zeros(4)}
            metric(pred_dict=pred_dict, target_dict=target_dict)
            
            print(metric.get_metric())
        except Exception as e:
            print(e)
            return
        self.assertTrue(True, False), "No exception catches."
    
    def test_AccuaryMetric4(self):
        # (5) check reset
        metric = AccuracyMetric()
        pred_dict = {"pred": torch.randn(4, 3, 2)}
        target_dict = {'target': torch.ones(4, 3)}
        metric(pred_dict=pred_dict, target_dict=target_dict)
        ans = torch.argmax(pred_dict["pred"], dim=2).to(target_dict["target"]) == target_dict["target"]
        res = metric.get_metric()
        self.assertTrue(isinstance(res, dict))
        self.assertTrue("acc" in res)
        self.assertAlmostEqual(res["acc"], float(ans.float().mean()), places=3)
    
    def test_AccuaryMetric5(self):
        # (5) check reset
        metric = AccuracyMetric()
        pred_dict = {"pred": torch.randn(4, 3, 2)}
        target_dict = {'target': torch.zeros(4, 3)}
        metric(pred_dict=pred_dict, target_dict=target_dict)
        res = metric.get_metric(reset=False)
        ans = (torch.argmax(pred_dict["pred"], dim=2).float() == target_dict["target"]).float().mean()
        self.assertAlmostEqual(res["acc"], float(ans), places=4)
    
    def test_AccuaryMetric6(self):
        # (6) check numpy array is not acceptable
        try:
            metric = AccuracyMetric()
            pred_dict = {"pred": np.zeros((4, 3, 2))}
            target_dict = {'target': np.zeros((4, 3))}
            metric(pred_dict=pred_dict, target_dict=target_dict)
        except Exception as e:
            print(e)
            return
        self.assertTrue(True, False), "No exception catches."
    
    def test_AccuaryMetric7(self):
        # (7) check map, match
        metric = AccuracyMetric(pred='predictions', target='targets')
        pred_dict = {"predictions": torch.randn(4, 3, 2)}
        target_dict = {'targets': torch.zeros(4, 3)}
        metric(pred_dict=pred_dict, target_dict=target_dict)
        res = metric.get_metric()
        ans = (torch.argmax(pred_dict["predictions"], dim=2).float() == target_dict["targets"]).float().mean()
        self.assertAlmostEqual(res["acc"], float(ans), places=4)
    
    def test_AccuaryMetric8(self):
        try:
            metric = AccuracyMetric(pred='predictions', target='targets')
            pred_dict = {"predictions": torch.zeros(4, 3, 2)}
            target_dict = {'targets': torch.zeros(4, 3)}
            metric(pred_dict=pred_dict, target_dict=target_dict, )
            self.assertDictEqual(metric.get_metric(), {'acc': 1})
        except Exception as e:
            print(e)
            return
        self.assertTrue(True, False), "No exception catches."
    
    def test_AccuaryMetric9(self):
        # (9) check map, include unused
        try:
            metric = AccuracyMetric(pred='prediction', target='targets')
            pred_dict = {"prediction": torch.zeros(4, 3, 2), 'unused': 1}
            target_dict = {'targets': torch.zeros(4, 3)}
            metric(pred_dict=pred_dict, target_dict=target_dict)
            self.assertDictEqual(metric.get_metric(), {'acc': 1})
        except Exception as e:
            print(e)
            return
        self.assertTrue(True, False), "No exception catches."
    
    def test_AccuaryMetric10(self):
        # (10) check _fast_metric
        try:
            metric = AccuracyMetric()
            pred_dict = {"predictions": torch.zeros(4, 3, 2), "seq_len": torch.ones(3) * 3}
            target_dict = {'targets': torch.zeros(4, 3)}
            metric(pred_dict=pred_dict, target_dict=target_dict)
            self.assertDictEqual(metric.get_metric(), {'acc': 1})
        except Exception as e:
            print(e)
            return
        self.assertTrue(True, False), "No exception catches."

    def test_duplicate(self):
        # 0.4.1的潜在bug，不能出现形参重复的情况
        metric = AccuracyMetric(pred='predictions', target='targets')
        pred_dict = {"predictions": torch.zeros(4, 3, 2), "seq_len": torch.ones(4) * 3, 'pred':0}
        target_dict = {'targets':torch.zeros(4, 3), 'target': 0}
        metric(pred_dict=pred_dict, target_dict=target_dict)


    def test_seq_len(self):
        N = 256
        seq_len = torch.zeros(N).long()
        seq_len[0] = 2
        pred = {'pred': torch.ones(N, 2)}
        target = {'target': torch.ones(N, 2), 'seq_len': seq_len}
        metric = AccuracyMetric()
        metric(pred_dict=pred, target_dict=target)
        self.assertDictEqual(metric.get_metric(), {'acc': 1.})
        seq_len[1:] = 1
        metric(pred_dict=pred, target_dict=target)
        self.assertDictEqual(metric.get_metric(), {'acc': 1.})


class SpanFPreRecMetricTest(unittest.TestCase):
    def test_case1(self):
        from fastNLP.core.metrics import _bmes_tag_to_spans
        from fastNLP.core.metrics import _bio_tag_to_spans
        
        bmes_lst = ['M-8', 'S-2', 'S-0', 'B-9', 'B-6', 'E-5', 'B-7', 'S-2', 'E-7', 'S-8']
        bio_lst = ['O-8', 'O-2', 'B-0', 'O-9', 'I-6', 'I-5', 'I-7', 'I-2', 'I-7', 'O-8']
        expect_bmes_res = set()
        expect_bmes_res.update([('8', (0, 1)), ('2', (1, 2)), ('0', (2, 3)), ('9', (3, 4)), ('6', (4, 5)),
                                ('5', (5, 6)), ('7', (6, 7)), ('2', (7, 8)), ('7', (8, 9)), ('8', (9, 10))])
        expect_bio_res = set()
        expect_bio_res.update([('7', (8, 9)), ('0', (2, 3)), ('2', (7, 8)), ('5', (5, 6)),
                               ('6', (4, 5)), ('7', (6, 7))])
        self.assertSetEqual(expect_bmes_res, set(_bmes_tag_to_spans(bmes_lst)))
        self.assertSetEqual(expect_bio_res, set(_bio_tag_to_spans(bio_lst)))

    def test_case2(self):
        # 测试不带label的
        from fastNLP.core.metrics import _bmes_tag_to_spans
        from fastNLP.core.metrics import _bio_tag_to_spans
        
        bmes_lst = ['B', 'E', 'B', 'S', 'B', 'M', 'E', 'M', 'B', 'E']
        bio_lst = ['I', 'B', 'O', 'O', 'I', 'O', 'I', 'B', 'O', 'O']
        expect_bmes_res = set()
        expect_bmes_res.update([('', (0, 2)), ('', (2, 3)), ('', (3, 4)), ('', (4, 7)), ('', (7, 8)), ('', (8, 10))])
        expect_bio_res = set()
        expect_bio_res.update([('', (7, 8)), ('', (6, 7)), ('', (4, 5)), ('', (0, 1)), ('', (1, 2))])
        self.assertSetEqual(expect_bmes_res, set(_bmes_tag_to_spans(bmes_lst)))
        self.assertSetEqual(expect_bio_res, set(_bio_tag_to_spans(bio_lst)))

    def test_case3(self):
        number_labels = 4
        # bio tag
        fastnlp_bio_vocab = Vocabulary(unknown=None, padding=None)
        fastnlp_bio_vocab.word_count = Counter(_generate_tags('BIO', number_labels))
        fastnlp_bio_metric = SpanFPreRecMetric(tag_vocab=fastnlp_bio_vocab, only_gross=False)
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
        bio_target = torch.LongTensor([[3, 6, 0, 8, 2, 4],
                                        [4, 1, 7, 0, 4, 7]])
        fastnlp_bio_metric({'pred': bio_sequence, 'seq_len': torch.LongTensor([6, 6])}, {'target': bio_target})
        expect_bio_res = {'pre-1': 0.333333, 'rec-1': 0.333333, 'f-1': 0.333333, 'pre-2': 0.5, 'rec-2': 0.5,
                          'f-2': 0.5, 'pre-0': 0.0, 'rec-0': 0.0, 'f-0': 0.0, 'pre-3': 0.0, 'rec-3': 0.0,
                          'f-3': 0.0, 'pre': 0.222222, 'rec': 0.181818, 'f': 0.2}

        self.assertDictEqual(expect_bio_res, fastnlp_bio_metric.get_metric())

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
                    if word_len==1:
                        target_i.append('S')
                    else:
                        target_i.append('B')
                        target_i.extend(['M']*(word_len-2))
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
                    if span[1]-span[0]>0:
                        tmp.extend(['B'] + ['M']*(span[1]-span[0]-1) + ['E'])
                    else:
                        tmp.append('S')
                pred_target.append(tmp)
            for i in range(len(seq_len)):
                raw_pred = pred_target[i]
                start = 0
                for j in range(seq_len[i]):
                    if raw_target[i][j] in ('E', 'S'):
                        flag = True
                        for k in range(start, j+1):
                            if raw_target[i][k]!=raw_pred[k]:
                                flag = False
                                break
                        if flag:
                            tp += 1
                        start = j + 1
                        gold += 1
                    if raw_pred[j] in ('E', 'S'):
                        seg += 1

            pre = round(tp/seg, 6)
            rec = round(tp/gold, 6)
            return {'f': round(2*pre*rec/(pre+rec), 6), 'pre': pre, 'rec':rec}

        target, raw_target, seq_len, vocab = _generate_samples()
        pred = torch.randn(3, max(seq_len), 4)

        expected_metric = get_eval(raw_target, pred, vocab, seq_len)
        metric = SpanFPreRecMetric(vocab, encoding_type='bmes')
        metric({'pred': pred, 'seq_len':torch.LongTensor(seq_len)}, {'target': torch.from_numpy(target)})
        # print(metric.get_metric(reset=False))
        # print(expected_metric)
        metric_value = metric.get_metric()
        for key, value in expected_metric.items():
            self.assertAlmostEqual(value, metric_value[key], places=5)

    def test_auto_encoding_type_infer(self):
        #  检查是否可以自动check encode的类型
        vocabs = {}
        import random
        for encoding_type in ['bio', 'bioes', 'bmeso']:
            vocab = Vocabulary(unknown=None, padding=None)
            for i in range(random.randint(10, 100)):
                label = str(random.randint(1, 10))
                for tag in encoding_type:
                    if tag!='o':
                        vocab.add_word(f'{tag}-{label}')
                    else:
                        vocab.add_word('o')
            vocabs[encoding_type] = vocab
        for e in ['bio', 'bioes', 'bmeso']:
            with self.subTest(e=e):
                metric = SpanFPreRecMetric(tag_vocab=vocabs[e])
                assert metric.encoding_type == e

        bmes_vocab = _generate_tags('bmes')
        vocab = Vocabulary()
        for tag, index in bmes_vocab.items():
            vocab.add_word(tag)
        metric = SpanFPreRecMetric(vocab)
        assert metric.encoding_type == 'bmes'

        # 一些无法check的情况
        vocab = Vocabulary()
        for i in range(10):
            vocab.add_word(str(i))
        with self.assertRaises(Exception):
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
                    if tag!='o':
                        vocab.add_word(f'{tag}-{label}')
                    else:
                        vocab.add_word('o')
            vocabs[encoding_type] = vocab
        for e1, e2 in product(['bio', 'bioes', 'bmeso'], ['bio', 'bioes', 'bmeso']):
            with self.subTest(e1=e1, e2=e2):
                if e1==e2:
                    metric = SpanFPreRecMetric(vocabs[e1], encoding_type=e2)
                else:
                    s2 = set(e2)
                    s2.update(set(e1))
                    if s2==set(e2):
                        continue
                    with self.assertRaises(AssertionError):
                        metric = SpanFPreRecMetric(vocabs[e1], encoding_type=e2)
        for encoding_type in ['bio', 'bioes', 'bmeso']:
            with self.assertRaises(AssertionError):
                metric = SpanFPreRecMetric(vocabs[encoding_type], encoding_type='bmes')

        with self.assertWarns(Warning):
            vocab = Vocabulary(unknown=None, padding=None).add_word_lst(list('bmes'))
            metric = SpanFPreRecMetric(vocab, encoding_type='bmeso')
            vocab = Vocabulary().add_word_lst(list('bmes'))
            metric = SpanFPreRecMetric(vocab, encoding_type='bmeso')


class TestCMRC2018Metric(unittest.TestCase):
    def test_case1(self):
        # 测试能否正确计算
        import torch
        metric = CMRC2018Metric()

        raw_chars = [list("abcsdef"), list("123456s789")]
        context_len = torch.LongTensor([3, 6])
        answers = [["abc", "abc", "abc"], ["12", "12", "12"]]
        pred_start = torch.randn(2, max(map(len, raw_chars)))
        pred_end = torch.randn(2, max(map(len, raw_chars)))
        pred_start[0, 0] = 1000  # 正好是abc
        pred_end[0, 2] = 1000
        pred_start[1, 1] = 1000  # 取出234
        pred_end[1, 3] = 1000

        metric.evaluate(answers=answers, raw_chars=raw_chars, pred_start=pred_start,
                        pred_end=pred_end, context_len=context_len)

        eval_res = metric.get_metric()
        self.assertDictEqual(eval_res, {'f1': 70.0, 'em': 50.0})


class TestUsefulFunctions(unittest.TestCase):
    # 测试metrics.py中一些看上去挺有用的函数
    def test_case_1(self):
        # multi-class
        _ = _accuracy_topk(np.random.randint(0, 3, size=(10, 1)), np.random.randint(0, 3, size=(10, 1)), k=3)
        _ = _pred_topk(np.random.randint(0, 3, size=(10, 1)))
        
        # 跑通即可



class TestClassfiyFPreRecMetric(unittest.TestCase):
    def test_case_1(self):
        pred= torch.tensor([[-0.4375, -0.1779, -1.0985, -1.1592,  0.4910],
        [ 1.3410,  0.2889, -0.8667, -1.8580,  0.3029],
        [ 0.7459, -1.1957,  0.3231,  0.0308, -0.1847],
        [ 1.1439, -0.0057,  0.8203,  0.0312, -1.0051],
        [-0.4870,  0.3215, -0.8290,  0.9221,  0.4683],
        [ 0.9078,  1.0674, -0.5629,  0.3895,  0.8917],
        [-0.7743, -0.4041, -0.9026,  0.2112,  1.0892],
        [ 1.8232, -1.4188, -2.5615, -2.4187,  0.5907],
        [-1.0592,  0.4164, -0.1192,  1.4238, -0.9258],
        [-1.1137,  0.5773,  2.5778,  0.5398, -0.3323],
        [-0.3868, -0.5165,  0.2286, -1.3876,  0.5561],
        [-0.3304,  1.3619, -1.5744,  0.4902, -0.7661],
        [ 1.8387,  0.5234,  0.4269,  1.3748, -1.2793],
        [ 0.6692,  0.2571,  1.2425, -0.5894, -0.0184],
        [ 0.4165,  0.4084, -0.1280,  1.4489, -2.3058],
        [-0.5826, -0.5469,  1.5898, -0.2786, -0.9882],
        [-1.5548, -2.2891,  0.2983, -1.2145, -0.1947],
        [-0.7222,  2.3543, -0.5801, -0.0640, -1.5614],
        [-1.4978,  1.9297, -1.3652, -0.2358,  2.5566],
        [ 0.1561, -0.0316,  0.9331,  1.0363,  2.3949],
        [ 0.2650, -0.8459,  1.3221,  0.1321, -1.1900],
        [ 0.0664, -1.2353, -0.5242, -1.4491,  1.3300],
        [-0.2744,  0.0941,  0.7157,  0.1404,  1.2046],
        [ 0.9341, -0.6652,  1.4512,  0.9608, -0.3623],
        [-1.1641,  0.0873,  0.1163, -0.2068, -0.7002],
        [ 1.4775, -2.0025, -0.5634, -0.1589,  0.0247],
        [ 1.0151,  1.0304, -0.1042, -0.6955, -0.0629],
        [-0.3119, -0.4558,  0.7757,  0.0758, -1.6297],
        [ 1.0654,  0.0313, -0.7716,  0.1194,  0.6913],
        [-0.8088, -0.6648, -0.5018, -0.0230, -0.8207],
        [-0.7753, -0.3508,  1.6163,  0.7158,  1.5207],
        [ 0.8692,  0.7718, -0.6734,  0.6515,  0.0641]])
        arg_max_pred = torch.argmax(pred,dim=-1)
        target = torch.tensor([0, 2, 4, 1, 4, 0, 1, 3, 3, 3, 1, 3, 4, 4, 3, 4, 0, 2, 4, 4, 3, 4, 4, 3,
        0, 3, 0, 0, 0, 1, 3, 1])
        
        metric = ClassifyFPreRecMetric(f_type='macro')
        metric.evaluate(pred, target)
        result_dict = metric.get_metric() 
        f1_score = 0.1882051282051282
        recall = 0.1619047619047619
        pre = 0.23928571428571427

        ground_truth = {'f': f1_score, 'pre': pre, 'rec': recall}
        for keys in ['f', 'pre', 'rec']:
            self.assertAlmostEqual(result_dict[keys], ground_truth[keys], delta=0.000001)

        metric = ClassifyFPreRecMetric(f_type='micro')
        metric.evaluate(pred, target)
        result_dict = metric.get_metric() 
        f1_score = 0.21875
        recall = 0.21875
        pre = 0.21875

        ground_truth = {'f': f1_score, 'pre': pre, 'rec': recall}
        for keys in ['f', 'pre', 'rec']:
            self.assertAlmostEqual(result_dict[keys], ground_truth[keys], delta=0.000001)

        metric = ClassifyFPreRecMetric(only_gross=False, f_type='macro')
        metric.evaluate(pred, target)
        result_dict = metric.get_metric(reset=True)
        ground_truth = {'0': {'f1-score': 0.13333333333333333, 'precision': 0.125, 'recall': 0.14285714285714285, 'support': 7}, '1': {'f1-score': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 5}, '2': {'f1-score': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 2}, '3': {'f1-score': 0.30769230769230765, 'precision': 0.5, 'recall': 0.2222222222222222, 'support': 9}, '4': {'f1-score': 0.5, 'precision': 0.5714285714285714, 'recall': 0.4444444444444444, 'support': 9}, 'macro avg': {'f1-score': 0.1882051282051282, 'precision': 0.23928571428571427, 'recall': 0.1619047619047619, 'support': 32}, 'micro avg': {'f1-score': 0.21875, 'precision': 0.21875, 'recall': 0.21875, 'support': 32}, 'weighted avg': {'f1-score': 0.2563301282051282, 'precision': 0.3286830357142857, 'recall': 0.21875, 'support': 32}}
        for keys in result_dict.keys():
            if keys=="f" or "pre" or "rec":
                continue
            gl=str(keys[-1])
            tmp_d={"p":"precision","r":"recall","f":"f1-score"}
            gk=tmp_d[keys[0]]
            self.assertAlmostEqual(result_dict[keys], ground_truth[gl][gk], delta=0.000001)

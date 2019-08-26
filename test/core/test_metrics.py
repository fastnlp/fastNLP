import unittest

import numpy as np
import torch

from fastNLP import AccuracyMetric
from fastNLP.core.metrics import _pred_topk, _accuracy_topk
from fastNLP.core.vocabulary import Vocabulary
from collections import Counter
from fastNLP.core.metrics import SpanFPreRecMetric, ExtractiveQAMetric


def _generate_tags(encoding_type, number_labels=4):
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
            pred_dict = {"prediction": torch.zeros(4, 3, 2)}
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


class SpanF1PreRecMetric(unittest.TestCase):
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


class TestUsefulFunctions(unittest.TestCase):
    # 测试metrics.py中一些看上去挺有用的函数
    def test_case_1(self):
        # multi-class
        _ = _accuracy_topk(np.random.randint(0, 3, size=(10, 1)), np.random.randint(0, 3, size=(10, 1)), k=3)
        _ = _pred_topk(np.random.randint(0, 3, size=(10, 1)))
        
        # 跑通即可


class TestExtractiveQAMetric(unittest.TestCase):

    def test_cast_1(self):
        qa_prediction = torch.FloatTensor([[[-0.4424, -0.4579, -0.7376, 1.8129, 0.1316, 1.6566, -1.2169,
                                            -0.3782, 0.8240],
                                           [-1.2348, -0.1876, -0.1462, -0.4834, -0.6692, -0.9735, -1.1563,
                                            -0.3562, -1.4116],
                                           [-1.6550, -0.9555, 0.3782, -1.3160, -1.5835, -0.3443, -1.7858,
                                            -2.0023, 0.0075],
                                           [-0.3772, -0.5447, -1.5631, 1.1614, 1.4598, -1.2764, 0.5186,
                                            0.3832, -0.1540],
                                           [-0.1011, 0.0600, 1.1090, -0.3545, 0.1284, 1.1484, -1.0120,
                                            -1.3508, -0.9513],
                                           [1.8948, 0.8627, -2.1359, 1.3740, -0.7499, 1.5019, 0.6919,
                                            -0.0842, -0.4294]],

                                          [[-0.2802, 0.6941, -0.4788, -0.3845, 1.7752, 1.2950, -1.9490,
                                            -1.4138, -0.8853],
                                           [-1.3752, -0.5457, -0.5305, 0.4018, 0.2934, 0.7931, 2.3845,
                                            -1.0726, 0.0364],
                                           [0.3621, 0.2609, 0.1269, -0.5950, 0.7212, 0.5959, 1.6264,
                                            -0.8836, -0.9320],
                                           [0.2003, -1.0758, -1.1560, -0.6472, -1.7549, 0.1264, 0.6044,
                                            -1.6857, 1.1571],
                                           [1.4277, -0.4915, 0.4496, 2.2027, 0.0730, -3.1792, -0.5125,
                                            3.5837, 1.0184],
                                           [1.6495, 1.7145, -0.2143, -0.1230, -0.2205, 0.8250, 0.4943,
                                            -0.9025, 0.0864]]])
        qa_prediction = qa_prediction.permute(1, 2, 0)
        pred1, pred2 = qa_prediction.split(1, dim=-1)
        pred1 = pred1.squeeze(-1)
        pred2 = pred2.squeeze(-1)
        target1 = torch.LongTensor([3, 0, 2, 4, 4, 0])
        target2 = torch.LongTensor([4, 1, 6, 8, 7, 1])
        metric = ExtractiveQAMetric()
        metric.evaluate(pred1, pred2, target1, target2)
        result = metric.get_metric()
        truth = {'EM': 62.5, 'f_1': 72.5, 'noAns-f_1': 50.0, 'noAns-EM': 50.0, 'hasAns-f_1': 95.0, 'hasAns-EM': 75.0}
        for k, v in truth.items():
            self.assertTrue(k in result)
            self.assertEqual(v, result[k])

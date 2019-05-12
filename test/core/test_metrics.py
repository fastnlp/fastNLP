import unittest

import numpy as np
import torch

from fastNLP import AccuracyMetric
from fastNLP import BMESF1PreRecMetric
from fastNLP.core.metrics import _pred_topk, _accuracy_topk


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
        # 已与allennlp对应函数做过验证，但由于测试不能依赖allennlp，所以这里只是截取上面的例子做固定测试
        # from allennlp.data.dataset_readers.dataset_utils import bio_tags_to_spans as allen_bio_tags_to_spans
        # from allennlp.data.dataset_readers.dataset_utils import bmes_tags_to_spans as allen_bmes_tags_to_spans
        # for i in range(1000):
        #     strs = list(map(str, np.random.randint(100, size=1000)))
        #     bmes = list('bmes'.upper())
        #     bmes_strs = [str_ + '-' + tag for tag, str_ in zip(strs, np.random.choice(bmes, size=len(strs)))]
        #     bio = list('bio'.upper())
        #     bio_strs = [str_ + '-' + tag for tag, str_ in zip(strs, np.random.choice(bio, size=len(strs)))]
        #     self.assertSetEqual(set(allen_bmes_tags_to_spans(bmes_strs)),set(bmes_tag_to_spans(bmes_strs)))
        #     self.assertSetEqual(set(allen_bio_tags_to_spans(bio_strs)), set(bio_tag_to_spans(bio_strs)))
    
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
        # 已与allennlp对应函数做过验证，但由于测试不能依赖allennlp，所以这里只是截取上面的例子做固定测试
        # from allennlp.data.dataset_readers.dataset_utils import bio_tags_to_spans as allen_bio_tags_to_spans
        # from allennlp.data.dataset_readers.dataset_utils import bmes_tags_to_spans as allen_bmes_tags_to_spans
        # for i in range(1000):
        #     bmes = list('bmes'.upper())
        #     bmes_strs = np.random.choice(bmes, size=1000)
        #     bio = list('bio'.upper())
        #     bio_strs = np.random.choice(bio, size=100)
        #     self.assertSetEqual(set(allen_bmes_tags_to_spans(bmes_strs)),set(bmes_tag_to_spans(bmes_strs)))
        #     self.assertSetEqual(set(allen_bio_tags_to_spans(bio_strs)), set(bio_tag_to_spans(bio_strs)))
    
    def tese_case3(self):
        from fastNLP.core.vocabulary import Vocabulary
        from collections import Counter
        from fastNLP.core.metrics import SpanFPreRecMetric
        # 与allennlp测试能否正确计算f metric
        #
        def generate_allen_tags(encoding_type, number_labels=4):
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
        
        number_labels = 4
        # bio tag
        fastnlp_bio_vocab = Vocabulary(unknown=None, padding=None)
        fastnlp_bio_vocab.word_count = Counter(generate_allen_tags('BIO', number_labels))
        fastnlp_bio_metric = SpanFPreRecMetric(tag_vocab=fastnlp_bio_vocab, only_gross=False)
        bio_sequence = torch.FloatTensor(
            [[[-0.9543, -1.4357, -0.2365, 0.2438, 1.0312, -1.4302, 0.3011,
               0.0470, 0.0971],
              [-0.6638, -0.7116, -1.9804, 0.2787, -0.2732, -0.9501, -1.4523,
               0.7987, -0.3970],
              [0.2939, 0.8132, -0.0903, -2.8296, 0.2080, -0.9823, -0.1898,
               0.6880, 1.4348],
              [-0.1886, 0.0067, -0.6862, -0.4635, 2.2776, 0.0710, -1.6793,
               -1.6876, -0.8917],
              [-0.7663, 0.6377, 0.8669, 0.1237, 1.7628, 0.0313, -1.0824,
               1.4217, 0.2622]],
            
             [[0.1529, 0.7474, -0.9037, 1.5287, 0.2771, 0.2223, 0.8136,
               1.3592, -0.8973],
              [0.4515, -0.5235, 0.3265, -1.1947, 0.8308, 1.8754, -0.4887,
               -0.4025, -0.3417],
              [-0.7855, 0.1615, -0.1272, -1.9289, -0.5181, 1.9742, -0.9698,
               0.2861, -0.3966],
              [-0.8291, -0.8823, -1.1496, 0.2164, 1.3390, -0.3964, -0.5275,
               0.0213, 1.4777],
              [-1.1299, 0.0627, -0.1358, -1.5951, 0.4484, -0.6081, -1.9566,
               1.3024, 0.2001]]]
        )
        bio_target = torch.LongTensor([[5., 0., 3., 3., 3.],
                                       [5., 6., 8., 6., 0.]])
        fastnlp_bio_metric({'pred': bio_sequence, 'seq_lens': torch.LongTensor([5, 5])}, {'target': bio_target})
        expect_bio_res = {'pre-1': 0.24999999999999373, 'rec-1': 0.499999999999975, 'f-1': 0.33333333333327775,
                          'pre-2': 0.0, 'rec-2': 0.0, 'f-2': 0.0, 'pre-3': 0.0, 'rec-3': 0.0, 'f-3': 0.0, 'pre-0': 0.0,
                          'rec-0': 0.0, 'f-0': 0.0, 'pre': 0.12499999999999845, 'rec': 0.12499999999999845,
                          'f': 0.12499999999994846}
        self.assertDictEqual(expect_bio_res, fastnlp_bio_metric.get_metric())
        
        # bmes tag
        bmes_sequence = torch.FloatTensor(
            [[[0.6536, -0.7179, 0.6579, 1.2503, 0.4176, 0.6696, 0.2352,
               -0.4085, 0.4084, -0.4185, 1.4172, -0.9162, -0.2679, 0.3332,
               -0.3505, -0.6002],
              [0.3238, -1.2378, -1.3304, -0.4903, 1.4518, -0.1868, -0.7641,
               1.6199, -0.8877, 0.1449, 0.8995, -0.5810, 0.1041, 0.1002,
               0.4439, 0.2514],
              [-0.8362, 2.9526, 0.8008, 0.1193, 1.0488, 0.6670, 1.1696,
               -1.1006, -0.8540, -0.1600, -0.9519, -0.2749, -0.4948, -1.4753,
               0.5802, -0.0516],
              [-0.8383, -1.7292, -1.4079, -1.5023, 0.5383, 0.6653, 0.3121,
               4.1249, -0.4173, -0.2043, 1.7755, 1.1110, -1.7069, -0.0390,
               -0.9242, -0.0333],
              [0.9088, -0.4955, -0.5076, 0.3732, 0.0283, -0.0263, -1.0393,
               0.7734, 1.0968, 0.4132, -1.3647, -0.5762, 0.6678, 0.8809,
               -0.3779, -0.3195]],
            
             [[-0.4638, -0.5939, -0.1052, -0.5573, 0.4600, -1.3484, 0.1753,
               0.0685, 0.3663, -0.6789, 0.0097, 1.0327, -0.0212, -0.9957,
               -0.1103, 0.4417],
              [-0.2903, 0.9205, -1.5758, -1.0421, 0.2921, -0.2142, -0.3049,
               -0.0879, -0.4412, -1.3195, -0.0657, -0.2986, 0.7214, 0.0631,
               -0.6386, 0.2797],
              [0.6440, -0.3748, 1.2912, -0.0170, 0.7447, 1.4075, -0.4947,
               0.4123, -0.8447, -0.5502, 0.3520, -0.2832, 0.5019, -0.1522,
               1.1237, -1.5385],
              [0.2839, -0.7649, 0.9067, -0.1163, -1.3789, 0.2571, -1.3977,
               -0.3680, -0.8902, -0.6983, -1.1583, 1.2779, 0.2197, 0.1376,
               -0.0591, -0.2461],
              [-0.2977, -1.8564, -0.5347, 1.0011, -1.1260, 0.4252, -2.0097,
               2.6973, -0.8308, -1.4939, 0.9865, -0.3935, 0.2743, 0.1142,
               -0.7344, -1.2046]]]
        )
        bmes_target = torch.LongTensor([[9., 6., 1., 9., 15.],
                                        [6., 15., 6., 15., 5.]])
        
        fastnlp_bmes_vocab = Vocabulary(unknown=None, padding=None)
        fastnlp_bmes_vocab.word_count = Counter(generate_allen_tags('BMES', number_labels))
        fastnlp_bmes_metric = SpanFPreRecMetric(tag_vocab=fastnlp_bmes_vocab, only_gross=False, encoding_type='bmes')
        fastnlp_bmes_metric({'pred': bmes_sequence, 'seq_lens': torch.LongTensor([20, 20])}, {'target': bmes_target})
        
        expect_bmes_res = {'f-3': 0.6666666666665778, 'pre-3': 0.499999999999975, 'rec-3': 0.9999999999999001,
                           'f-0': 0.0, 'pre-0': 0.0, 'rec-0': 0.0, 'f-1': 0.33333333333327775,
                           'pre-1': 0.24999999999999373, 'rec-1': 0.499999999999975, 'f-2': 0.7499999999999314,
                           'pre-2': 0.7499999999999812, 'rec-2': 0.7499999999999812, 'f': 0.49999999999994504,
                           'pre': 0.499999999999995, 'rec': 0.499999999999995}
        
        self.assertDictEqual(fastnlp_bmes_metric.get_metric(), expect_bmes_res)
        
        # 已经和allennlp做过验证，但由于不能依赖allennlp，所以注释了以下代码
        # from allennlp.data.vocabulary import Vocabulary  as allen_Vocabulary
        # from allennlp.training.metrics import SpanBasedF1Measure
        # allen_bio_vocab = allen_Vocabulary({"tags": generate_allen_tags('BIO', number_labels)},
        #                                    non_padded_namespaces=['tags'])
        # allen_bio_metric = SpanBasedF1Measure(allen_bio_vocab, 'tags')
        # bio_sequence = torch.randn(size=(2, 20, 2 * number_labels + 1))
        # bio_target = torch.randint(2 * number_labels + 1, size=(2, 20))
        # allen_bio_metric(bio_sequence, bio_target, torch.ones(2, 20))
        # fastnlp_bio_vocab = Vocabulary(unknown=None, padding=None)
        # fastnlp_bio_vocab.word_count = Counter(generate_allen_tags('BIO', number_labels))
        # fastnlp_bio_metric = SpanFPreRecMetric(tag_vocab=fastnlp_bio_vocab, only_gross=False)
        #
        # def convert_allen_res_to_fastnlp_res(metric_result):
        #     allen_result = {}
        #     key_map = {'f1-measure-overall': "f", "recall-overall": "rec", "precision-overall": "pre"}
        #     for key, value in metric_result.items():
        #         if key in key_map:
        #             key = key_map[key]
        #         else:
        #             label = key.split('-')[-1]
        #             if key.startswith('f1'):
        #                 key = 'f-{}'.format(label)
        #             else:
        #                 key = '{}-{}'.format(key[:3], label)
        #         allen_result[key] = value
        #     return allen_result
        #
        # # print(convert_allen_res_to_fastnlp_res(allen_bio_metric.get_metric()))
        # # print(fastnlp_bio_metric.get_metric())
        # self.assertDictEqual(convert_allen_res_to_fastnlp_res(allen_bio_metric.get_metric()),
        #                      fastnlp_bio_metric.get_metric())
        #
        # allen_bmes_vocab = allen_Vocabulary({"tags": generate_allen_tags('BMES', number_labels)})
        # allen_bmes_metric = SpanBasedF1Measure(allen_bmes_vocab, 'tags', label_encoding='BMES')
        # fastnlp_bmes_vocab = Vocabulary(unknown=None, padding=None)
        # fastnlp_bmes_vocab.word_count = Counter(generate_allen_tags('BMES', number_labels))
        # fastnlp_bmes_metric = SpanFPreRecMetric(tag_vocab=fastnlp_bmes_vocab, only_gross=False, encoding_type='bmes')
        # bmes_sequence = torch.randn(size=(2, 20, 4 * number_labels))
        # bmes_target = torch.randint(4 * number_labels, size=(2, 20))
        # allen_bmes_metric(bmes_sequence, bmes_target, torch.ones(2, 20))
        # fastnlp_bmes_metric({'pred': bmes_sequence, 'seq_lens': torch.LongTensor([20, 20])}, {'target': bmes_target})
        #
        # # print(convert_allen_res_to_fastnlp_res(allen_bmes_metric.get_metric()))
        # # print(fastnlp_bmes_metric.get_metric())
        # self.assertDictEqual(convert_allen_res_to_fastnlp_res(allen_bmes_metric.get_metric()),
        #                      fastnlp_bmes_metric.get_metric())



class TestUsefulFunctions(unittest.TestCase):
    # 测试metrics.py中一些看上去挺有用的函数
    def test_case_1(self):
        # multi-class
        _ = _accuracy_topk(np.random.randint(0, 3, size=(10, 1)), np.random.randint(0, 3, size=(10, 1)), k=3)
        _ = _pred_topk(np.random.randint(0, 3, size=(10, 1)))
        
        # 跑通即可

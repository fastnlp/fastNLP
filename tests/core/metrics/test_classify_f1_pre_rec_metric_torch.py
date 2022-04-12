from functools import partial
import copy

import pytest
import torch
import numpy as np
from torch.multiprocessing import Pool, set_start_method

from fastNLP.core.metrics import ClassifyFPreRecMetric
from fastNLP.core.dataset import DataSet
from .utils import find_free_network_port, setup_ddp

set_start_method("spawn", force=True)


def _test(local_rank: int, world_size: int, device: torch.device,
          dataset: DataSet, metric_class, metric_kwargs, metric_result):
    metric = metric_class(**metric_kwargs)
    # dataset 也类似（每个进程有自己的一个）
    dataset = copy.deepcopy(dataset)
    metric.to(device)
    # 把数据拆到每个 GPU 上，有点模仿 DistributedSampler 的感觉，但这里数据单位是一个 batch（即每个 i 取了一个 batch 到自己的 GPU 上）
    for i in range(local_rank, len(dataset), world_size):
        pred, tg = dataset[i]['pred'].to(device), dataset[i]['tg'].to(device)
        metric.update(pred, tg)

    my_result = metric.get_metric()
    for keys in ['f', 'pre', 'rec']:
        np.allclose(my_result[keys], metric_result[keys], atol=0.000001)


class TestClassfiyFPreRecMetric:
    def test_case_1(self):
        pred = torch.tensor([[-0.4375, -0.1779, -1.0985, -1.1592, 0.4910],
                             [1.3410, 0.2889, -0.8667, -1.8580, 0.3029],
                             [0.7459, -1.1957, 0.3231, 0.0308, -0.1847],
                             [1.1439, -0.0057, 0.8203, 0.0312, -1.0051],
                             [-0.4870, 0.3215, -0.8290, 0.9221, 0.4683],
                             [0.9078, 1.0674, -0.5629, 0.3895, 0.8917],
                             [-0.7743, -0.4041, -0.9026, 0.2112, 1.0892],
                             [1.8232, -1.4188, -2.5615, -2.4187, 0.5907],
                             [-1.0592, 0.4164, -0.1192, 1.4238, -0.9258],
                             [-1.1137, 0.5773, 2.5778, 0.5398, -0.3323],
                             [-0.3868, -0.5165, 0.2286, -1.3876, 0.5561],
                             [-0.3304, 1.3619, -1.5744, 0.4902, -0.7661],
                             [1.8387, 0.5234, 0.4269, 1.3748, -1.2793],
                             [0.6692, 0.2571, 1.2425, -0.5894, -0.0184],
                             [0.4165, 0.4084, -0.1280, 1.4489, -2.3058],
                             [-0.5826, -0.5469, 1.5898, -0.2786, -0.9882],
                             [-1.5548, -2.2891, 0.2983, -1.2145, -0.1947],
                             [-0.7222, 2.3543, -0.5801, -0.0640, -1.5614],
                             [-1.4978, 1.9297, -1.3652, -0.2358, 2.5566],
                             [0.1561, -0.0316, 0.9331, 1.0363, 2.3949],
                             [0.2650, -0.8459, 1.3221, 0.1321, -1.1900],
                             [0.0664, -1.2353, -0.5242, -1.4491, 1.3300],
                             [-0.2744, 0.0941, 0.7157, 0.1404, 1.2046],
                             [0.9341, -0.6652, 1.4512, 0.9608, -0.3623],
                             [-1.1641, 0.0873, 0.1163, -0.2068, -0.7002],
                             [1.4775, -2.0025, -0.5634, -0.1589, 0.0247],
                             [1.0151, 1.0304, -0.1042, -0.6955, -0.0629],
                             [-0.3119, -0.4558, 0.7757, 0.0758, -1.6297],
                             [1.0654, 0.0313, -0.7716, 0.1194, 0.6913],
                             [-0.8088, -0.6648, -0.5018, -0.0230, -0.8207],
                             [-0.7753, -0.3508, 1.6163, 0.7158, 1.5207],
                             [0.8692, 0.7718, -0.6734, 0.6515, 0.0641]])
        arg_max_pred = torch.argmax(pred, dim=-1)
        target = torch.tensor([0, 2, 4, 1, 4, 0, 1, 3, 3, 3, 1, 3, 4, 4, 3, 4, 0, 2, 4, 4, 3, 4, 4, 3,
                               0, 3, 0, 0, 0, 1, 3, 1])

        metric = ClassifyFPreRecMetric(f_type='macro', num_class=5)
        metric.update(pred, target)
        result_dict = metric.get_metric()
        f1_score = 0.1882051282051282
        recall = 0.1619047619047619
        pre = 0.23928571428571427

        ground_truth = {'f': f1_score, 'pre': pre, 'rec': recall}
        for keys in ['f', 'pre', 'rec']:
            np.allclose(result_dict[keys], ground_truth[keys], atol=0.000001)

        metric = ClassifyFPreRecMetric(f_type='micro', num_class=5)
        metric.update(pred, target)
        result_dict = metric.get_metric()
        f1_score = 0.21875
        recall = 0.21875
        pre = 0.21875

        ground_truth = {'f': f1_score, 'pre': pre, 'rec': recall}
        for keys in ['f', 'pre', 'rec']:
            np.allclose(result_dict[keys], ground_truth[keys], atol=0.000001)

        metric = ClassifyFPreRecMetric(only_gross=False, f_type='macro', num_class=5)
        metric.update(pred, target)
        result_dict = metric.get_metric()
        ground_truth = {
            '0': {'f1-score': 0.13333333333333333, 'precision': 0.125, 'recall': 0.14285714285714285, 'support': 7},
            '1': {'f1-score': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 5},
            '2': {'f1-score': 0.0, 'precision': 0.0, 'recall': 0.0, 'support': 2},
            '3': {'f1-score': 0.30769230769230765, 'precision': 0.5, 'recall': 0.2222222222222222, 'support': 9},
            '4': {'f1-score': 0.5, 'precision': 0.5714285714285714, 'recall': 0.4444444444444444, 'support': 9},
            'macro avg': {'f1-score': 0.1882051282051282, 'precision': 0.23928571428571427,
                          'recall': 0.1619047619047619, 'support': 32},
            'micro avg': {'f1-score': 0.21875, 'precision': 0.21875, 'recall': 0.21875, 'support': 32},
            'weighted avg': {'f1-score': 0.2563301282051282, 'precision': 0.3286830357142857, 'recall': 0.21875,
                             'support': 32}}
        for keys in result_dict.keys():
            if keys == "f" or "pre" or "rec":
                continue
            gl = str(keys[-1])
            tmp_d = {"p": "precision", "r": "recall", "f": "f1-score"}
            gk = tmp_d[keys[0]]
            np.allclose(result_dict[keys], ground_truth[gl][gk], atol=0.000001)

    @pytest.mark.parametrize("f_type, f1_score,recall,pre",
                             [('macro', 0.1882051282051282, 0.1619047619047619, 0.23928571428571427),
                              ('micro', 0.21875, 0.21875, 0.21875)])
    def test_case_2(self, f_type, f1_score, recall, pre):
        dataset = DataSet({
            'pred': [torch.tensor([[-0.4375, -0.1779, -1.0985, -1.1592, 0.4910],
                                   [1.3410, 0.2889, -0.8667, -1.8580, 0.3029],
                                   [0.7459, -1.1957, 0.3231, 0.0308, -0.1847],
                                   [1.1439, -0.0057, 0.8203, 0.0312, -1.0051],
                                   [-0.4870, 0.3215, -0.8290, 0.9221, 0.4683],
                                   [0.9078, 1.0674, -0.5629, 0.3895, 0.8917],
                                   [-0.7743, -0.4041, -0.9026, 0.2112, 1.0892],
                                   [1.8232, -1.4188, -2.5615, -2.4187, 0.5907],
                                   [-1.0592, 0.4164, -0.1192, 1.4238, -0.9258],
                                   [-1.1137, 0.5773, 2.5778, 0.5398, -0.3323],
                                   [-0.3868, -0.5165, 0.2286, -1.3876, 0.5561],
                                   [-0.3304, 1.3619, -1.5744, 0.4902, -0.7661],
                                   [1.8387, 0.5234, 0.4269, 1.3748, -1.2793],
                                   [0.6692, 0.2571, 1.2425, -0.5894, -0.0184],
                                   [0.4165, 0.4084, -0.1280, 1.4489, -2.3058],
                                   [-0.5826, -0.5469, 1.5898, -0.2786, -0.9882]]),
                     torch.tensor([
                         [-1.5548, -2.2891, 0.2983, -1.2145, -0.1947],
                         [-0.7222, 2.3543, -0.5801, -0.0640, -1.5614],
                         [-1.4978, 1.9297, -1.3652, -0.2358, 2.5566],
                         [0.1561, -0.0316, 0.9331, 1.0363, 2.3949],
                         [0.2650, -0.8459, 1.3221, 0.1321, -1.1900],
                         [0.0664, -1.2353, -0.5242, -1.4491, 1.3300],
                         [-0.2744, 0.0941, 0.7157, 0.1404, 1.2046],
                         [0.9341, -0.6652, 1.4512, 0.9608, -0.3623],
                         [-1.1641, 0.0873, 0.1163, -0.2068, -0.7002],
                         [1.4775, -2.0025, -0.5634, -0.1589, 0.0247],
                         [1.0151, 1.0304, -0.1042, -0.6955, -0.0629],
                         [-0.3119, -0.4558, 0.7757, 0.0758, -1.6297],
                         [1.0654, 0.0313, -0.7716, 0.1194, 0.6913],
                         [-0.8088, -0.6648, -0.5018, -0.0230, -0.8207],
                         [-0.7753, -0.3508, 1.6163, 0.7158, 1.5207],
                         [0.8692, 0.7718, -0.6734, 0.6515, 0.0641]
                     ])],
            'tg': [
                torch.LongTensor([0, 2, 4, 1, 4, 0, 1, 3, 3, 3, 1, 3, 4, 4, 3, 4]),
                torch.LongTensor([0, 2, 4, 4, 3, 4, 4, 3, 0, 3, 0, 0, 0, 1, 3, 1])
            ]
        })
        metric_kwargs = {
            'f_type': f_type,
            'num_class': 5,
            'only_gross': False,
            'aggregate_when_get_metric': True
        }
        ground_truth = {'f': f1_score, 'pre': pre, 'rec': recall}

        NUM_PROCESSES = 2
        pool = Pool(processes=NUM_PROCESSES)
        master_port = find_free_network_port()
        pool.starmap(setup_ddp, [(rank, NUM_PROCESSES, master_port) for rank in range(NUM_PROCESSES)])

        pool.starmap(partial(_test, dataset=dataset,
                             metric_class=ClassifyFPreRecMetric,
                             metric_kwargs=metric_kwargs,
                             metric_result=ground_truth),
                     [(rank, NUM_PROCESSES, torch.device(f'cuda:{rank+4}')) for rank in range(NUM_PROCESSES)])
        pool.close()
        pool.join()

import unittest
import _pickle
from fastNLP import cache_results
from fastNLP.io import EmbedLoader
from fastNLP import DataSet
from fastNLP import Instance
import time
import os
import torch
from torch import nn
from fastNLP.core.utils import _move_model_to_device, _get_model_device
import numpy as np
from fastNLP.core.utils import seq_len_to_mask

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(0))


class TestMoveModelDevice(unittest.TestCase):
    def test_case1(self):
        # 测试str
        model = Model()
        model = _move_model_to_device(model, 'cpu')
        assert model.param.device == torch.device('cpu')
        # 测试不存在的device报错
        with self.assertRaises(Exception):
            _move_model_to_device(model, 'cpuu')
        # 测试gpu
        if torch.cuda.is_available():
            model = _move_model_to_device(model, 'cuda')
            assert model.param.is_cuda
            model = _move_model_to_device(model, 'cuda:0')
            assert model.param.device == torch.device('cuda:0')
            with self.assertRaises(Exception):
                _move_model_to_device(model, 'cuda:1000')
        # 测试None
        model = _move_model_to_device(model, None)
    
    def test_case2(self):
        # 测试使用int初始化
        model = Model()
        if torch.cuda.is_available():
            model = _move_model_to_device(model, 0)
            assert model.param.device == torch.device('cuda:0')
            assert model.param.device == torch.device('cuda:0'), "The model should be in "
            with self.assertRaises(Exception):
                _move_model_to_device(model, 100)
            with self.assertRaises(Exception):
                _move_model_to_device(model, -1)
    
    def test_case3(self):
        # 测试None
        model = Model()
        device = _get_model_device(model)
        model = _move_model_to_device(model, None)
        assert device == _get_model_device(model), "The device should not change."
        if torch.cuda.is_available():
            model.cuda()
            device = _get_model_device(model)
            model = _move_model_to_device(model, None)
            assert device == _get_model_device(model), "The device should not change."
            
            model = nn.DataParallel(model, device_ids=[0])
            _move_model_to_device(model, None)
            with self.assertRaises(Exception):
                _move_model_to_device(model, 'cpu')
    
    def test_case4(self):
        # 测试传入list的内容
        model = Model()
        device = ['cpu']
        with self.assertRaises(Exception):
            _move_model_to_device(model, device)
        if torch.cuda.is_available():
            device = [0]
            _model = _move_model_to_device(model, device)
            assert not isinstance(_model, nn.DataParallel)
            device = [torch.device('cuda:0'), torch.device('cuda:0')]
            with self.assertRaises(Exception):
                _model = _move_model_to_device(model, device)
            if torch.cuda.device_count() > 1:
                device = [0, 1]
                _model = _move_model_to_device(model, device)
                assert isinstance(_model, nn.DataParallel)
                device = ['cuda', 'cuda:1']
                with self.assertRaises(Exception):
                    _move_model_to_device(model, device)
    
    def test_case5(self):
        if not torch.cuda.is_available():
            return
        # torch.device()
        device = torch.device('cpu')
        model = Model()
        _move_model_to_device(model, device)
        device = torch.device('cuda')
        model = _move_model_to_device(model, device)
        assert model.param.device == torch.device('cuda:0')
        with self.assertRaises(Exception):
            _move_model_to_device(model, torch.device('cuda:100'))


@cache_results('test/demo1.pkl')
def process_data_1(embed_file, cws_train):
    embed, vocab = EmbedLoader.load_without_vocab(embed_file)
    time.sleep(1)  # 测试是否通过读取cache获得结果
    with open(cws_train, 'r', encoding='utf-8') as f:
        d = DataSet()
        for line in f:
            line = line.strip()
            if len(line) > 0:
                d.append(Instance(raw=line))
    return embed, vocab, d


class TestCache(unittest.TestCase):
    def test_cache_save(self):
        try:
            start_time = time.time()
            embed, vocab, d = process_data_1('test/data_for_tests/word2vec_test.txt', 'test/data_for_tests/cws_train')
            end_time = time.time()
            pre_time = end_time - start_time
            with open('test/demo1.pkl', 'rb') as f:
                _embed, _vocab, _d = _pickle.load(f)
            self.assertEqual(embed.shape, _embed.shape)
            for i in range(embed.shape[0]):
                self.assertListEqual(embed[i].tolist(), _embed[i].tolist())
            start_time = time.time()
            embed, vocab, d = process_data_1('test/data_for_tests/word2vec_test.txt', 'test/data_for_tests/cws_train')
            end_time = time.time()
            read_time = end_time - start_time
            print("Read using {:.3f}, while prepare using:{:.3f}".format(read_time, pre_time))
            self.assertGreater(pre_time - 0.5, read_time)
        finally:
            os.remove('test/demo1.pkl')
    
    def test_cache_save_overwrite_path(self):
        try:
            start_time = time.time()
            embed, vocab, d = process_data_1('test/data_for_tests/word2vec_test.txt', 'test/data_for_tests/cws_train',
                                             _cache_fp='test/demo_overwrite.pkl')
            end_time = time.time()
            pre_time = end_time - start_time
            with open('test/demo_overwrite.pkl', 'rb') as f:
                _embed, _vocab, _d = _pickle.load(f)
            self.assertEqual(embed.shape, _embed.shape)
            for i in range(embed.shape[0]):
                self.assertListEqual(embed[i].tolist(), _embed[i].tolist())
            start_time = time.time()
            embed, vocab, d = process_data_1('test/data_for_tests/word2vec_test.txt', 'test/data_for_tests/cws_train',
                                             _cache_fp='test/demo_overwrite.pkl')
            end_time = time.time()
            read_time = end_time - start_time
            print("Read using {:.3f}, while prepare using:{:.3f}".format(read_time, pre_time))
            self.assertGreater(pre_time - 0.5, read_time)
        finally:
            os.remove('test/demo_overwrite.pkl')
    
    def test_cache_refresh(self):
        try:
            start_time = time.time()
            embed, vocab, d = process_data_1('test/data_for_tests/word2vec_test.txt', 'test/data_for_tests/cws_train',
                                             _refresh=True)
            end_time = time.time()
            pre_time = end_time - start_time
            with open('test/demo1.pkl', 'rb') as f:
                _embed, _vocab, _d = _pickle.load(f)
            self.assertEqual(embed.shape, _embed.shape)
            for i in range(embed.shape[0]):
                self.assertListEqual(embed[i].tolist(), _embed[i].tolist())
            start_time = time.time()
            embed, vocab, d = process_data_1('test/data_for_tests/word2vec_test.txt', 'test/data_for_tests/cws_train',
                                             _refresh=True)
            end_time = time.time()
            read_time = end_time - start_time
            print("Read using {:.3f}, while prepare using:{:.3f}".format(read_time, pre_time))
            self.assertGreater(0.1, pre_time - read_time)
        finally:
            os.remove('test/demo1.pkl')
    
    def test_duplicate_keyword(self):
        with self.assertRaises(RuntimeError):
            @cache_results(None)
            def func_verbose(a, _verbose):
                pass
            
            func_verbose(0, 1)
        with self.assertRaises(RuntimeError):
            @cache_results(None)
            def func_cache(a, _cache_fp):
                pass
            
            func_cache(1, 2)
        with self.assertRaises(RuntimeError):
            @cache_results(None)
            def func_refresh(a, _refresh):
                pass
            
            func_refresh(1, 2)
    
    def test_create_cache_dir(self):
        @cache_results('test/demo1/demo.pkl')
        def cache():
            return 1, 2
        
        try:
            results = cache()
            print(results)
        finally:
            os.remove('test/demo1/demo.pkl')
            os.rmdir('test/demo1')


class TestSeqLenToMask(unittest.TestCase):

    def evaluate_mask_seq_len(self, seq_len, mask):
        max_len = int(max(seq_len))
        for i in range(len(seq_len)):
            length = seq_len[i]
            mask_i = mask[i]
            for j in range(max_len):
                self.assertEqual(mask_i[j], j<length)

    def test_numpy_seq_len(self):
        # 测试能否转换numpy类型的seq_len
        # 1. 随机测试
        seq_len = np.random.randint(1, 10, size=(10, ))
        mask = seq_len_to_mask(seq_len)
        max_len = seq_len.max()
        self.assertEqual(max_len, mask.shape[1])
        self.evaluate_mask_seq_len(seq_len, mask)

        # 2. 异常检测
        seq_len = np.random.randint(10, size=(10, 1))
        with self.assertRaises(AssertionError):
            mask = seq_len_to_mask(seq_len)


    def test_pytorch_seq_len(self):
        # 1. 随机测试
        seq_len = torch.randint(1, 10, size=(10, ))
        max_len = seq_len.max()
        mask = seq_len_to_mask(seq_len)
        self.assertEqual(max_len, mask.shape[1])
        self.evaluate_mask_seq_len(seq_len.tolist(), mask)

        # 2. 异常检测
        seq_len = torch.randn(3, 4)
        with self.assertRaises(AssertionError):
            mask = seq_len_to_mask(seq_len)


import unittest
import _pickle
from fastNLP import cache_results
from fastNLP.io.embed_loader import EmbedLoader
from fastNLP import DataSet
from fastNLP import Instance
import time
import os

@cache_results('test/demo1.pkl')
def process_data_1(embed_file, cws_train):
    embed, vocab = EmbedLoader.load_without_vocab(embed_file)
    time.sleep(1)  # 测试是否通过读取cache获得结果
    with open(cws_train, 'r', encoding='utf-8') as f:
        d = DataSet()
        for line in f:
            line = line.strip()
            if len(line)>0:
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
            self.assertGreater(pre_time-0.5, read_time)
        finally:
            os.remove('test/demo1.pkl')

    def test_cache_save_overwrite_path(self):
        try:
            start_time = time.time()
            embed, vocab, d = process_data_1('test/data_for_tests/word2vec_test.txt', 'test/data_for_tests/cws_train',
                                             cache_filepath='test/demo_overwrite.pkl')
            end_time = time.time()
            pre_time = end_time - start_time
            with open('test/demo_overwrite.pkl', 'rb') as f:
                _embed, _vocab, _d = _pickle.load(f)
            self.assertEqual(embed.shape, _embed.shape)
            for i in range(embed.shape[0]):
                self.assertListEqual(embed[i].tolist(), _embed[i].tolist())
            start_time = time.time()
            embed, vocab, d = process_data_1('test/data_for_tests/word2vec_test.txt', 'test/data_for_tests/cws_train',
                                             cache_filepath='test/demo_overwrite.pkl')
            end_time = time.time()
            read_time = end_time - start_time
            print("Read using {:.3f}, while prepare using:{:.3f}".format(read_time, pre_time))
            self.assertGreater(pre_time-0.5, read_time)
        finally:
            os.remove('test/demo_overwrite.pkl')

    def test_cache_refresh(self):
        try:
            start_time = time.time()
            embed, vocab, d = process_data_1('test/data_for_tests/word2vec_test.txt', 'test/data_for_tests/cws_train',
                                             refresh=True)
            end_time = time.time()
            pre_time = end_time - start_time
            with open('test/demo1.pkl', 'rb') as f:
                _embed, _vocab, _d = _pickle.load(f)
            self.assertEqual(embed.shape, _embed.shape)
            for i in range(embed.shape[0]):
                self.assertListEqual(embed[i].tolist(), _embed[i].tolist())
            start_time = time.time()
            embed, vocab, d = process_data_1('test/data_for_tests/word2vec_test.txt', 'test/data_for_tests/cws_train',
                                             refresh=True)
            end_time = time.time()
            read_time = end_time - start_time
            print("Read using {:.3f}, while prepare using:{:.3f}".format(read_time, pre_time))
            self.assertGreater(0.1, pre_time-read_time)
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
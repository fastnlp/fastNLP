
import unittest
import os
from fastNLP.io.pipe.cws import CWSPipe


class TestCWSPipe(unittest.TestCase):
    @unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
    def test_process_from_file(self):
        dataset_names = ['pku', 'cityu', 'as', 'msra']
        for dataset_name in dataset_names:
            with self.subTest(dataset_name=dataset_name):
                data_bundle = CWSPipe(dataset_name=dataset_name).process_from_file()
                print(data_bundle)

    def test_demo(self):
        # related to issue https://github.com/fastnlp/fastNLP/issues/324#issue-705081091
        from fastNLP import DataSet, Instance
        from fastNLP.io import DataBundle
        data_bundle = DataBundle()
        ds = DataSet()
        ds.append(Instance(raw_words="截流 进入 最后 冲刺 （ 附 图片 １ 张 ）"))
        data_bundle.set_dataset(ds, name='train')
        data_bundle = CWSPipe().process(data_bundle)
        self.assertFalse('<' in data_bundle.get_vocab('chars'))


class TestRunCWSPipe(unittest.TestCase):
    def test_process_from_file(self):
        dataset_names = ['msra', 'cityu', 'as', 'pku']
        for dataset_name in dataset_names:
            with self.subTest(dataset_name=dataset_name):
                data_bundle = CWSPipe(bigrams=True, trigrams=True).\
                    process_from_file(f'test/data_for_tests/io/cws_{dataset_name}')
                print(data_bundle)

    def test_replace_number(self):
        data_bundle = CWSPipe(bigrams=True, replace_num_alpha=True).\
                    process_from_file(f'test/data_for_tests/io/cws_pku')
        for word in ['<', '>', '<NUM>']:
            self.assertNotEqual(data_bundle.get_vocab('chars').to_index(word), 1)

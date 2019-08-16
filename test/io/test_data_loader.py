import unittest

from fastNLP.core.const import Const
from fastNLP.io.data_loader import MNLILoader


class TestDataLoader(unittest.TestCase):

    def test_mnli_loader(self):
        ds = MNLILoader().process('test/data_for_tests/sample_mnli.tsv',
                                  to_lower=True, get_index=True, seq_len_type='mask')
        self.assertTrue('train' in ds.datasets)
        self.assertTrue(len(ds.datasets) == 1)
        self.assertTrue(len(ds.datasets['train']) == 11)
        self.assertTrue(isinstance(ds.datasets['train'][0][Const.INPUT_LENS(0)], list))

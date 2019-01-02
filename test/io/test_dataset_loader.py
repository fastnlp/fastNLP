import os
import unittest

from fastNLP.io.dataset_loader import Conll2003Loader
class TestDatasetLoader(unittest.TestCase):
    
    def test_case_1(self):
        '''
            Test the the loader of Conll2003 dataset
        '''

        dataset_path = "test/data_for_tests/conll_2003_example.txt"
        loader = Conll2003Loader()
        dataset_2003 = loader.load(dataset_path)
        
        for item in dataset_2003:
            len0 = len(item["label0_list"])
            len1 = len(item["label1_list"])
            len2 = len(item["label2_list"])
            lentoken = len(item["token_list"])
            self.assertNotEqual(len0, 0)
            self.assertEqual(len0, len1)
            self.assertEqual(len1, len2)
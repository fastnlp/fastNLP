import unittest
from ..data_load.cr_loader import CRLoader

class Test_CRLoader(unittest.TestCase):
    def test_cr_loader(self):
        train_path = 'data/train.english.jsonlines.mini'
        dev_path = 'data/dev.english.jsonlines.minid'
        test_path = 'data/test.english.jsonlines'
        cr = CRLoader()
        data_info = cr.process({'train':train_path,'dev':dev_path,'test':test_path})

        print(data_info.datasets['train'][0])
        print(data_info.datasets['dev'][0])
        print(data_info.datasets['test'][0])

from fastNLP.io.loader.coreference import CRLoader
import unittest

class TestCR(unittest.TestCase):
    def test_load(self):

        test_root = "../../data_for_tests/coreference/"
        train_path = test_root+"coreference_train.json"
        dev_path = test_root+"coreference_dev.json"
        test_path = test_root+"coreference_test.json"
        paths = {"train": train_path,"dev":dev_path,"test":test_path}

        bundle1 = CRLoader().load(paths)
        bundle2 = CRLoader().load(test_root)
        print(bundle1)
        print(bundle2)
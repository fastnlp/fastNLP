from fastNLP.io.loader.coreference import CoReferenceLoader
import unittest


class TestCR(unittest.TestCase):
    def test_load(self):

        test_root = "test/data_for_tests/io/coreference/"
        train_path = test_root+"coreference_train.json"
        dev_path = test_root+"coreference_dev.json"
        test_path = test_root+"coreference_test.json"
        paths = {"train": train_path, "dev": dev_path, "test": test_path}

        bundle1 = CoReferenceLoader().load(paths)
        bundle2 = CoReferenceLoader().load(test_root)
        print(bundle1)
        print(bundle2)

        self.assertEqual(bundle1.num_dataset, 3)
        self.assertEqual(bundle2.num_dataset, 3)
        self.assertEqual(bundle1.num_vocab, 0)
        self.assertEqual(bundle2.num_vocab, 0)

        self.assertEqual(len(bundle1.get_dataset('train')), 1)
        self.assertEqual(len(bundle1.get_dataset('dev')), 1)
        self.assertEqual(len(bundle1.get_dataset('test')), 1)

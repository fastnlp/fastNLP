import unittest
from fastNLP.io.pipe.coreference import CoReferencePipe


class TestCR(unittest.TestCase):

    def test_load(self):
        class Config():
            max_sentences = 50
            filter = [3, 4, 5]
            char_path = None
        config = Config()

        file_root_path = "tests/data_for_tests/io/coreference/"
        train_path = file_root_path + "coreference_train.json"
        dev_path = file_root_path + "coreference_dev.json"
        test_path = file_root_path + "coreference_test.json"

        paths = {"train": train_path, "dev": dev_path, "test": test_path}

        bundle1 = CoReferencePipe(config).process_from_file(paths)
        bundle2 = CoReferencePipe(config).process_from_file(file_root_path)
        print(bundle1)
        print(bundle2)
        self.assertEqual(bundle1.num_dataset, 3)
        self.assertEqual(bundle2.num_dataset, 3)
        self.assertEqual(bundle1.num_vocab, 1)
        self.assertEqual(bundle2.num_vocab, 1)

        self.assertEqual(len(bundle1.get_dataset('train')), 1)
        self.assertEqual(len(bundle1.get_dataset('dev')), 1)
        self.assertEqual(len(bundle1.get_dataset('test')), 1)
        self.assertEqual(len(bundle1.get_vocab('words1')), 84)

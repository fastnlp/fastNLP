import unittest
from fastNLP.io.pipe.coreference import CoreferencePipe


class TestCR(unittest.TestCase):

    def test_load(self):
        class Config():
            max_sentences = 50
            filter = [3, 4, 5]
            char_path = None
        config = Config()

        file_root_path = "../../data_for_tests/coreference/"
        train_path = file_root_path + "coreference_train.json"
        dev_path = file_root_path + "coreference_dev.json"
        test_path = file_root_path + "coreference_test.json"

        paths = {"train": train_path, "dev": dev_path, "test": test_path}

        bundle1 = CoreferencePipe(config).process_from_file(paths)
        bundle2 = CoreferencePipe(config).process_from_file(file_root_path)
        print(bundle1)
        print(bundle2)
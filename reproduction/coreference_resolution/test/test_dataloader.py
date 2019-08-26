

import unittest
from fastNLP.io.pipe.coreference import CoreferencePipe
from reproduction.coreference_resolution.model.config import Config

class Test_CRLoader(unittest.TestCase):
    def test_cr_loader(self):
        config = Config()
        bundle = CoreferencePipe(config).process_from_file({'train': config.train_path, 'dev': config.dev_path,'test': config.test_path})

        print(bundle.datasets['train'][0])
        print(bundle.datasets['dev'][0])
        print(bundle.datasets['test'][0])

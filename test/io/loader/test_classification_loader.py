
import unittest

import os

from fastNLP.io import DataBundle
from fastNLP.io.loader.classification import YelpFullLoader, YelpPolarityLoader, IMDBLoader, \
    SSTLoader, SST2Loader, ChnSentiCorpLoader, THUCNewsLoader, WeiboSenti100kLoader


@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
class TestDownload(unittest.TestCase):
    def test_download(self):
        for loader in [YelpFullLoader, YelpPolarityLoader, IMDBLoader, SST2Loader, SSTLoader, ChnSentiCorpLoader]:
            loader().download()

    def test_load(self):
        for loader in [YelpFullLoader, YelpPolarityLoader, IMDBLoader, SST2Loader, SSTLoader, ChnSentiCorpLoader]:
            data_bundle = loader().load()
            print(data_bundle)


class TestLoad(unittest.TestCase):
    def test_process_from_file(self):
        data_set_dict = {
            'yelp.p': ('test/data_for_tests/io/yelp_review_polarity', YelpPolarityLoader, (6, 6, 6), False),
            'yelp.f': ('test/data_for_tests/io/yelp_review_full', YelpFullLoader, (6, 6, 6), False),
            'sst-2': ('test/data_for_tests/io/SST-2', SST2Loader, (5, 5, 5), True),
            'sst': ('test/data_for_tests/io/SST', SSTLoader, (6, 6, 6), False),
            'imdb': ('test/data_for_tests/io/imdb', IMDBLoader, (6, 6, 6), False),
            'ChnSentiCorp': ('test/data_for_tests/io/ChnSentiCorp', ChnSentiCorpLoader, (6, 6, 6), False),
            'THUCNews': ('test/data_for_tests/io/THUCNews', THUCNewsLoader, (9, 9, 9), False),
            'WeiboSenti100k': ('test/data_for_tests/io/WeiboSenti100k', WeiboSenti100kLoader, (7, 6, 6), False),
        }
        for k, v in data_set_dict.items():
            path, loader, data_set, warns = v
            with self.subTest(loader=loader):
                if warns:
                    with self.assertWarns(Warning):
                        data_bundle = loader().load(path)
                else:
                    data_bundle = loader().load(path)

                self.assertTrue(isinstance(data_bundle, DataBundle))
                self.assertEqual(len(data_set), data_bundle.num_dataset)
                for x, y in zip(data_set, data_bundle.iter_datasets()):
                    name, dataset = y
                    self.assertEqual(x, len(dataset))


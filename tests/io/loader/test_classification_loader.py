

import os
import pytest

from fastNLP.io import DataBundle
from fastNLP.io.loader.classification import YelpFullLoader, YelpPolarityLoader, IMDBLoader, \
    SSTLoader, SST2Loader, ChnSentiCorpLoader, THUCNewsLoader, WeiboSenti100kLoader, \
    MRLoader, R8Loader, R52Loader, OhsumedLoader, NG20Loader


class TestDownload:
    @pytest.mark.skipif('download' not in os.environ, reason="Skip download")
    def test_download(self):
        for loader in [YelpFullLoader, YelpPolarityLoader, IMDBLoader, SST2Loader, SSTLoader, ChnSentiCorpLoader]:
            loader().download()

    @pytest.mark.skipif('download' not in os.environ, reason="Skip download")
    def test_load(self):
        for loader in [YelpFullLoader, YelpPolarityLoader, IMDBLoader, SST2Loader, SSTLoader, ChnSentiCorpLoader]:
            data_bundle = loader().load()
            print(data_bundle)


class TestLoad:
    def test_process_from_file(self):
        data_set_dict = {
            'yelp.p': ('data_for_tests/io/yelp_review_polarity', YelpPolarityLoader, (6, 6, 6), False),
            'yelp.f': ('data_for_tests/io/yelp_review_full', YelpFullLoader, (6, 6, 6), False),
            'sst-2': ('data_for_tests/io/SST-2', SST2Loader, (5, 5, 5), True),
            'sst': ('data_for_tests/io/SST', SSTLoader, (6, 6, 6), False),
            'imdb': ('data_for_tests/io/imdb', IMDBLoader, (6, 6, 6), False),
            'ChnSentiCorp': ('data_for_tests/io/ChnSentiCorp', ChnSentiCorpLoader, (6, 6, 6), False),
            'THUCNews': ('data_for_tests/io/THUCNews', THUCNewsLoader, (9, 9, 9), False),
            'WeiboSenti100k': ('data_for_tests/io/WeiboSenti100k', WeiboSenti100kLoader, (6, 7, 6), False),
            'mr': ('data_for_tests/io/mr', MRLoader, (6, 6, 6), False),
            'R8': ('data_for_tests/io/R8', R8Loader, (6, 6, 6), False),
            'R52': ('data_for_tests/io/R52', R52Loader, (6, 6, 6), False),
            'ohsumed': ('data_for_tests/io/R52', OhsumedLoader, (6, 6, 6), False),
            '20ng': ('data_for_tests/io/R52', NG20Loader, (6, 6, 6), False),
        }
        for k, v in data_set_dict.items():
            path, loader, data_set, warns = v
            data_bundle = loader().load(path)

            assert(isinstance(data_bundle, DataBundle))
            assert(len(data_set) == data_bundle.num_dataset)
            for x, y in zip(data_set, data_bundle.iter_datasets()):
                name, dataset = y
                assert(x == len(dataset))


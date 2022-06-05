import pytest
import os
from fastNLP.io.loader import CWSLoader


class TestCWSLoader:
    @pytest.mark.skipif('download' not in os.environ, reason="Skip download")
    def test_download(self):
        dataset_names = ['pku', 'cityu', 'as', 'msra']
        for dataset_name in dataset_names:
            data_bundle = CWSLoader(dataset_name=dataset_name).load()
            print(data_bundle)


class TestRunCWSLoader:
    def test_cws_loader(self):
        dataset_names = ['msra', 'cityu', 'as', 'msra']
        for dataset_name in dataset_names:
            data_bundle = CWSLoader(dataset_name=dataset_name).load(
                f'tests/data_for_tests/io/cws_{dataset_name}'
            )
            print(data_bundle)


import pytest

import os

from fastNLP.io import DataBundle
from fastNLP.io.loader.matching import RTELoader, QNLILoader, SNLILoader, QuoraLoader, MNLILoader, \
    BQCorpusLoader, CNXNLILoader, LCQMCLoader


@pytest.mark.skipif('download' not in os.environ, reason="Skip download")
class TestMatchingDownload:
    def test_download(self):
        for loader in [RTELoader, QNLILoader, SNLILoader, MNLILoader]:
            loader().download()
        with pytest.raises(Exception):
            QuoraLoader().load()

    def test_load(self):
        for loader in [RTELoader, QNLILoader, SNLILoader, MNLILoader]:
            data_bundle = loader().load()
            print(data_bundle)


class TestMatchingLoad:
    def test_load(self):
        data_set_dict = {
            'RTE': ('data_for_tests/io/RTE', RTELoader, (5, 5, 5), True),
            'SNLI': ('data_for_tests/io/SNLI', SNLILoader, (5, 5, 5), False),
            'QNLI': ('data_for_tests/io/QNLI', QNLILoader, (5, 5, 5), True),
            'MNLI': ('data_for_tests/io/MNLI', MNLILoader, (5, 5, 5, 5, 6), True),
            'Quora': ('data_for_tests/io/Quora', QuoraLoader, (2, 2, 2), False),
            'BQCorpus': ('data_for_tests/io/BQCorpus', BQCorpusLoader, (5, 5, 5), False),
            'XNLI': ('data_for_tests/io/XNLI', CNXNLILoader, (6, 6, 8), False),
            'LCQMC': ('data_for_tests/io/LCQMC', LCQMCLoader, (6, 5, 6), False),
        }
        for k, v in data_set_dict.items():
            path, loader, instance, warns = v
            if warns:
                data_bundle = loader().load(path)
            else:
                data_bundle = loader().load(path)

            assert(isinstance(data_bundle, DataBundle))
            assert(len(instance) == data_bundle.num_dataset)
            for x, y in zip(instance, data_bundle.iter_datasets()):
                name, dataset = y
                assert(x == len(dataset))


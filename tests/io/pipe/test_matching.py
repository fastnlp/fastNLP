
import pytest
import os

from fastNLP.io import DataBundle
from fastNLP.io.pipe.matching import SNLIPipe, RTEPipe, QNLIPipe, QuoraPipe, MNLIPipe, \
    CNXNLIPipe, BQCorpusPipe, LCQMCPipe
from fastNLP.io.pipe.matching import SNLIBertPipe, RTEBertPipe, QNLIBertPipe, QuoraBertPipe, MNLIBertPipe, \
    CNXNLIBertPipe, BQCorpusBertPipe, LCQMCBertPipe


@pytest.mark.skipif('download' not in os.environ, reason="Skip download")
class TestMatchingPipe:
    def test_process_from_file(self):
        for pipe in [SNLIPipe, RTEPipe, QNLIPipe, MNLIPipe]:
            print(pipe)
            data_bundle = pipe(tokenizer='raw').process_from_file()
            print(data_bundle)


@pytest.mark.skipif('download' not in os.environ, reason="Skip download")
class TestMatchingBertPipe:
    def test_process_from_file(self):
        for pipe in [SNLIBertPipe, RTEBertPipe, QNLIBertPipe, MNLIBertPipe]:
            print(pipe)
            data_bundle = pipe(tokenizer='raw').process_from_file()
            print(data_bundle)


class TestRunMatchingPipe:
    def test_load(self):
        data_set_dict = {
            'RTE': ('data_for_tests/io/RTE', RTEPipe, RTEBertPipe, (5, 5, 5), (449, 2), True),
            'SNLI': ('data_for_tests/io/SNLI', SNLIPipe, SNLIBertPipe, (5, 5, 5), (110, 3), False),
            'QNLI': ('data_for_tests/io/QNLI', QNLIPipe, QNLIBertPipe, (5, 5, 5), (372, 2), True),
            'MNLI': ('data_for_tests/io/MNLI', MNLIPipe, MNLIBertPipe, (5, 5, 5, 5, 6), (459, 3), True),
            'BQCorpus': ('data_for_tests/io/BQCorpus', BQCorpusPipe, BQCorpusBertPipe, (5, 5, 5), (32, 2), False),
            'XNLI': ('data_for_tests/io/XNLI', CNXNLIPipe, CNXNLIBertPipe, (6, 6, 8), (39, 3), False),
            'LCQMC': ('data_for_tests/io/LCQMC', LCQMCPipe, LCQMCBertPipe, (6, 5, 6), (36, 2), False),
        }
        for k, v in data_set_dict.items():
            path, pipe1, pipe2, data_set, vocab, warns = v
            if warns:
                data_bundle1 = pipe1(tokenizer='raw').process_from_file(path)
                data_bundle2 = pipe2(tokenizer='raw').process_from_file(path)
            else:
                data_bundle1 = pipe1(tokenizer='raw').process_from_file(path)
                data_bundle2 = pipe2(tokenizer='raw').process_from_file(path)

            assert(isinstance(data_bundle1, DataBundle))
            assert(len(data_set) == data_bundle1.num_dataset)
            print(k)
            print(data_bundle1)
            print(data_bundle2)
            for x, y in zip(data_set, data_bundle1.iter_datasets()):
                name, dataset = y
                assert(x == len(dataset))
            assert(len(data_set) == data_bundle2.num_dataset)
            for x, y in zip(data_set, data_bundle2.iter_datasets()):
                name, dataset = y
                assert(x == len(dataset))

            assert(len(vocab) == data_bundle1.num_vocab)
            for x, y in zip(vocab, data_bundle1.iter_vocabs()):
                name, vocabs = y
                assert(x == len(vocabs))
            assert(len(vocab) == data_bundle2.num_vocab)
            for x, y in zip(vocab, data_bundle1.iter_vocabs()):
                name, vocabs = y
                assert(x + 1 if name == 'words' else x == len(vocabs))

    def test_load_proc(self):
        data_set_dict = {
            'RTE': ('data_for_tests/io/RTE', RTEPipe, RTEBertPipe, (5, 5, 5), (449, 2), True),
            'SNLI': ('data_for_tests/io/SNLI', SNLIPipe, SNLIBertPipe, (5, 5, 5), (110, 3), False),
            'QNLI': ('data_for_tests/io/QNLI', QNLIPipe, QNLIBertPipe, (5, 5, 5), (372, 2), True),
            'MNLI': ('data_for_tests/io/MNLI', MNLIPipe, MNLIBertPipe, (5, 5, 5, 5, 6), (459, 3), True),
            'BQCorpus': ('data_for_tests/io/BQCorpus', BQCorpusPipe, BQCorpusBertPipe, (5, 5, 5), (32, 2), False),
            'XNLI': ('data_for_tests/io/XNLI', CNXNLIPipe, CNXNLIBertPipe, (6, 6, 8), (39, 3), False),
            'LCQMC': ('data_for_tests/io/LCQMC', LCQMCPipe, LCQMCBertPipe, (6, 5, 6), (36, 2), False),
        }
        for k, v in data_set_dict.items():
            path, pipe1, pipe2, data_set, vocab, warns = v
            if warns:
                data_bundle1 = pipe1(tokenizer='raw', num_proc=2).process_from_file(path)
                data_bundle2 = pipe2(tokenizer='raw', num_proc=2).process_from_file(path)
            else:
                data_bundle1 = pipe1(tokenizer='raw', num_proc=2).process_from_file(path)
                data_bundle2 = pipe2(tokenizer='raw', num_proc=2).process_from_file(path)

            assert (isinstance(data_bundle1, DataBundle))
            assert (len(data_set) == data_bundle1.num_dataset)
            print(k)
            print(data_bundle1)
            print(data_bundle2)
            for x, y in zip(data_set, data_bundle1.iter_datasets()):
                name, dataset = y
                assert (x == len(dataset))
            assert (len(data_set) == data_bundle2.num_dataset)
            for x, y in zip(data_set, data_bundle2.iter_datasets()):
                name, dataset = y
                assert (x == len(dataset))

            assert (len(vocab) == data_bundle1.num_vocab)
            for x, y in zip(vocab, data_bundle1.iter_vocabs()):
                name, vocabs = y
                assert (x == len(vocabs))
            assert (len(vocab) == data_bundle2.num_vocab)
            for x, y in zip(vocab, data_bundle1.iter_vocabs()):
                name, vocabs = y
                assert (x + 1 if name == 'words' else x == len(vocabs))

    @pytest.mark.skipif('download' not in os.environ, reason="Skip download")
    def test_spacy(self):
        data_set_dict = {
            'Quora': ('data_for_tests/io/Quora', QuoraPipe, QuoraBertPipe, (2, 2, 2), (93, 2)),
            }
        for k, v in data_set_dict.items():
            path, pipe1, pipe2, data_set, vocab = v

            data_bundle1 = pipe1(tokenizer='spacy').process_from_file(path)
            data_bundle2 = pipe2(tokenizer='spacy').process_from_file(path)

            assert(isinstance(data_bundle1, DataBundle))
            assert(len(data_set) == data_bundle1.num_dataset)
            print(k)
            print(data_bundle1)
            print(data_bundle2)
            for x, y in zip(data_set, data_bundle1.iter_datasets()):
                name, dataset = y
                assert(x == len(dataset))
            assert(len(data_set) == data_bundle2.num_dataset)
            for x, y in zip(data_set, data_bundle2.iter_datasets()):
                name, dataset = y
                assert(x == len(dataset))

            assert(len(vocab) == data_bundle1.num_vocab)
            for x, y in zip(vocab, data_bundle1.iter_vocabs()):
                name, vocabs = y
                assert(x == len(vocabs))
            assert(len(vocab) == data_bundle2.num_vocab)
            for x, y in zip(vocab, data_bundle1.iter_vocabs()):
                name, vocabs = y
                assert(x + 1 if name == 'words' else x == len(vocabs))


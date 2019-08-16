import unittest
import os

from fastNLP.io.pipe.classification import SSTPipe, SST2Pipe, IMDBPipe, YelpFullPipe, YelpPolarityPipe

@unittest.skipIf('TRAVIS' in os.environ, "Skip in travis")
class TestPipe(unittest.TestCase):
    def test_process_from_file(self):
        for pipe in [YelpPolarityPipe, SST2Pipe, IMDBPipe, YelpFullPipe,  SSTPipe]:
            with self.subTest(pipe=pipe):
                print(pipe)
                data_bundle = pipe(tokenizer='raw').process_from_file()
                print(data_bundle)

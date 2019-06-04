

import unittest
from reproduction.seqence_labelling.cws.data.CWSDataLoader import SigHanLoader
from fastNLP.core.vocabulary import VocabularyOption


class TestCWSDataLoader(unittest.TestCase):
    def test_case1(self):
        cws_loader = SigHanLoader(target_type='bmes')
        data = cws_loader.process('pku_demo.txt')
        print(data.datasets)

    def test_calse2(self):
        cws_loader = SigHanLoader(target_type='bmes')
        data = cws_loader.process('pku_demo.txt', bigram_vocab_opt=VocabularyOption())
        print(data.datasets)
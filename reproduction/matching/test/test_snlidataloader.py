import unittest
from ..data import MatchingDataLoader
from fastNLP.core.vocabulary import Vocabulary


class TestCWSDataLoader(unittest.TestCase):
    def test_case1(self):
        snli_loader = MatchingDataLoader()
        # TODO: still in progress


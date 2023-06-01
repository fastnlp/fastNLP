import pytest
from collections import Counter

from fastNLP.core.dataset import DataSet
from fastNLP.core.vocabulary import Vocabulary
from fastNLP import logger


class TestVocabulary:

    def test_from_dataset(self):
        ds = DataSet({"x": [[1, 2], [3, 4]], "y": ["apple", ""]})
        vocab = Vocabulary()
        vocab.from_dataset(ds, field_name="y")
        assert vocab.word_count == Counter({'apple': 1})

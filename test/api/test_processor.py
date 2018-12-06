import unittest

from fastNLP.api.processor import FullSpaceToHalfSpaceProcessor
from fastNLP.core.dataset import DataSet


class TestProcessor(unittest.TestCase):
    def test_FullSpaceToHalfSpaceProcessor(self):
        ds = DataSet({"word": ["０0, u１, u), (u２, u2"]})
        proc = FullSpaceToHalfSpaceProcessor("word")
        ds = proc(ds)
        self.assertTrue(ds.field_arrays["word"].content, ["00, u1, u), (u2, u2"])

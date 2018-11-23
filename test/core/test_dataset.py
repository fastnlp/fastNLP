import unittest

from fastNLP.core.dataset import DataSet


class TestDataSet(unittest.TestCase):

    def test_case_1(self):
        ds = DataSet()
        ds.add_field(name="xx", fields=["a", "b", "e", "d"])

        self.assertTrue("xx" in ds.field_arrays)
        self.assertEqual(len(ds.field_arrays["xx"]), 4)
        self.assertEqual(ds.get_length(), 4)
        self.assertEqual(ds.get_fields(), ds.field_arrays)

        try:
            ds.add_field(name="yy", fields=["x", "y", "z", "w", "f"])
        except BaseException as e:
            self.assertTrue(isinstance(e, AssertionError))

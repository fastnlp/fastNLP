import unittest

from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance


class TestDataSet(unittest.TestCase):

    def test_init_v1(self):
        ds = DataSet([Instance(x=[1, 2, 3, 4], y=[5, 6])] * 40)
        self.assertTrue("x" in ds.field_arrays and "y" in ds.field_arrays)
        self.assertEqual(ds.field_arrays["x"].content, [[1, 2, 3, 4], ] * 40)
        self.assertEqual(ds.field_arrays["y"].content, [[5, 6], ] * 40)

    def test_init_v2(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        self.assertTrue("x" in ds.field_arrays and "y" in ds.field_arrays)
        self.assertEqual(ds.field_arrays["x"].content, [[1, 2, 3, 4], ] * 40)
        self.assertEqual(ds.field_arrays["y"].content, [[5, 6], ] * 40)

    def test_init_assert(self):
        with self.assertRaises(AssertionError):
            _ = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 100})
        with self.assertRaises(AssertionError):
            _ = DataSet([[1, 2, 3, 4]] * 10)
        with self.assertRaises(ValueError):
            _ = DataSet(0.00001)

    def test_append(self):
        dd = DataSet()
        for _ in range(3):
            dd.append(Instance(x=[1, 2, 3, 4], y=[5, 6]))
        self.assertEqual(len(dd), 3)
        self.assertEqual(dd.field_arrays["x"].content, [[1, 2, 3, 4]] * 3)
        self.assertEqual(dd.field_arrays["y"].content, [[5, 6]] * 3)

    def test_add_append(self):
        dd = DataSet()
        dd.add_field("x", [[1, 2, 3]] * 10)
        dd.add_field("y", [[1, 2, 3, 4]] * 10)
        dd.add_field("z", [[5, 6]] * 10)
        self.assertEqual(len(dd), 10)
        self.assertEqual(dd.field_arrays["x"].content, [[1, 2, 3]] * 10)
        self.assertEqual(dd.field_arrays["y"].content, [[1, 2, 3, 4]] * 10)
        self.assertEqual(dd.field_arrays["z"].content, [[5, 6]] * 10)

    def test_delete_field(self):
        dd = DataSet()
        dd.add_field("x", [[1, 2, 3]] * 10)
        dd.add_field("y", [[1, 2, 3, 4]] * 10)
        dd.delete_field("x")
        self.assertFalse("x" in dd.field_arrays)
        self.assertTrue("y" in dd.field_arrays)

    def test_getitem(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        ins_1, ins_0 = ds[0], ds[1]
        self.assertTrue(isinstance(ins_1, Instance) and isinstance(ins_0, Instance))
        self.assertEqual(ins_1["x"], [1, 2, 3, 4])
        self.assertEqual(ins_1["y"], [5, 6])
        self.assertEqual(ins_0["x"], [1, 2, 3, 4])
        self.assertEqual(ins_0["y"], [5, 6])

        sub_ds = ds[:10]
        self.assertTrue(isinstance(sub_ds, DataSet))
        self.assertEqual(len(sub_ds), 10)

    def test_apply(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        ds.apply(lambda ins: ins["x"][::-1], new_field_name="rx")
        self.assertTrue("rx" in ds.field_arrays)
        self.assertEqual(ds.field_arrays["rx"].content[0], [4, 3, 2, 1])

import unittest

import numpy as np

from fastNLP.core.fieldarray import FieldArray


class TestFieldArray(unittest.TestCase):
    def test(self):
        fa = FieldArray("x", [1, 2, 3, 4, 5], is_input=True)
        self.assertEqual(len(fa), 5)
        fa.append(6)
        self.assertEqual(len(fa), 6)

        self.assertEqual(fa[-1], 6)
        self.assertEqual(fa[0], 1)
        fa[-1] = 60
        self.assertEqual(fa[-1], 60)

        self.assertEqual(fa.get(0), 1)
        self.assertTrue(isinstance(fa.get([0, 1, 2]), np.ndarray))
        self.assertListEqual(list(fa.get([0, 1, 2])), [1, 2, 3])

    def test_type_conversion(self):
        fa = FieldArray("x", [1.2, 2.2, 3, 4, 5], is_input=True)
        self.assertEqual(fa.pytype, float)
        self.assertEqual(fa.dtype, np.float64)

        fa = FieldArray("x", [1, 2, 3, 4, 5], is_input=True)
        fa.append(1.3333)
        self.assertEqual(fa.pytype, float)
        self.assertEqual(fa.dtype, np.float64)

        fa = FieldArray("y", [1.1, 2.2, 3.3, 4.4, 5.5], is_input=False)
        fa.append(10)
        self.assertEqual(fa.pytype, float)
        self.assertEqual(fa.dtype, np.float64)

        fa = FieldArray("y", ["a", "b", "c", "d"], is_input=False)
        fa.append("e")
        self.assertEqual(fa.dtype, np.str)
        self.assertEqual(fa.pytype, str)

    def test_support_np_array(self):
        fa = FieldArray("y", [np.array([1.1, 2.2, 3.3, 4.4, 5.5])], is_input=False)
        self.assertEqual(fa.dtype, np.ndarray)
        self.assertEqual(fa.pytype, np.ndarray)

        fa.append(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
        self.assertEqual(fa.dtype, np.ndarray)
        self.assertEqual(fa.pytype, np.ndarray)

        fa = FieldArray("my_field", np.random.rand(3, 5), is_input=False)
        # in this case, pytype is actually a float. We do not care about it.
        self.assertEqual(fa.dtype, np.float64)

    def test_nested_list(self):
        fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1.1, 2.2, 3.3, 4.4, 5.5]], is_input=False)
        self.assertEqual(fa.pytype, float)
        self.assertEqual(fa.dtype, np.float64)

    def test_getitem_v1(self):
        fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1, 2, 3, 4, 5]], is_input=True)
        self.assertEqual(fa[0], [1.1, 2.2, 3.3, 4.4, 5.5])
        ans = fa[[0, 1]]
        self.assertTrue(isinstance(ans, np.ndarray))
        self.assertTrue(isinstance(ans[0], np.ndarray))
        self.assertEqual(ans[0].tolist(), [1.1, 2.2, 3.3, 4.4, 5.5])
        self.assertEqual(ans[1].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(ans.dtype, np.float64)

    def test_getitem_v2(self):
        x = np.random.rand(10, 5)
        fa = FieldArray("my_field", x, is_input=True)
        indices = [0, 1, 3, 4, 6]
        for a, b in zip(fa[indices], x[indices]):
            self.assertListEqual(a.tolist(), b.tolist())

    def test_append(self):
        with self.assertRaises(Exception):
            fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1, 2, 3, 4, 5]], is_input=True)
            fa.append(0)

        with self.assertRaises(Exception):
            fa = FieldArray("y", [1.1, 2.2, 3.3, 4.4, 5.5], is_input=True)
            fa.append([1, 2, 3, 4, 5])

        with self.assertRaises(Exception):
            fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1, 2, 3, 4, 5]], is_input=True)
            fa.append([])

        with self.assertRaises(Exception):
            fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1, 2, 3, 4, 5]], is_input=True)
            fa.append(["str", 0, 0, 0, 1.89])

        fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1, 2, 3, 4, 5]], is_input=True)
        fa.append([1.2, 2.3, 3.4, 4.5, 5.6])
        self.assertEqual(len(fa), 3)
        self.assertEqual(fa[2], [1.2, 2.3, 3.4, 4.5, 5.6])

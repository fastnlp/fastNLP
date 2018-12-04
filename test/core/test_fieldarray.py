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

        fa.append(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
        self.assertEqual(fa.pytype, np.ndarray)

    def test_nested_list(self):
        fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1.1, 2.2, 3.3, 4.4, 5.5]], is_input=False)
        self.assertEqual(fa.pytype, float)
        self.assertEqual(fa.dtype, np.float64)

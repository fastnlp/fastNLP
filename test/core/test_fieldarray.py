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

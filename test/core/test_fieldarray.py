import unittest

import numpy as np

from fastNLP.core.fieldarray import FieldArray


class TestFieldArrayInit(unittest.TestCase):
    """
    1） 如果DataSet使用dict初始化，那么在add_field中会构造FieldArray：
            1.1) 二维list  DataSet({"x": [[1, 2], [3, 4]]})
            1.2) 二维array  DataSet({"x": np.array([[1, 2], [3, 4]])})
            1.3) 三维list  DataSet({"x": [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]})
        2） 如果DataSet使用list of Instance 初始化,那么在append中会先对第一个样本初始化FieldArray；
        然后后面的样本使用FieldArray.append进行添加。
            2.1) 一维list DataSet([Instance(x=[1, 2, 3, 4])])
            2.2) 一维array DataSet([Instance(x=np.array([1, 2, 3, 4]))])
            2.3) 二维list  DataSet([Instance(x=[[1, 2], [3, 4]])])
            2.4) 二维array  DataSet([Instance(x=np.array([[1, 2], [3, 4]]))])
    """

    def test_init_v1(self):
        # 二维list
        fa = FieldArray("x", [[1, 2], [3, 4]] * 5, is_input=True)

    def test_init_v2(self):
        # 二维array
        fa = FieldArray("x", np.array([[1, 2], [3, 4]] * 5), is_input=True)

    def test_init_v3(self):
        # 三维list
        fa = FieldArray("x", [[[1, 2], [3, 4]], [[1, 2], [3, 4]]], is_input=True)

    def test_init_v7(self):
        # list of array
        fa = FieldArray("x", [np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])], is_input=True)
        self.assertEqual(fa.pytype, int)
        self.assertEqual(fa.dtype, np.int)

    def test_init_v4(self):
        # 一维list
        val = [1, 2, 3, 4]
        fa = FieldArray("x", [val], is_input=True)
        fa.append(val)

    def test_init_v5(self):
        # 一维array
        val = np.array([1, 2, 3, 4])
        fa = FieldArray("x", [val], is_input=True)
        fa.append(val)

    def test_init_v6(self):
        # 二维array
        val = [[1, 2], [3, 4]]
        fa = FieldArray("x", [val], is_input=True)
        fa.append(val)

    def test_init_v7(self):
        # 二维list
        val = np.array([[1, 2], [3, 4]])
        fa = FieldArray("x", [val], is_input=True)
        fa.append(val)


class TestFieldArray(unittest.TestCase):
    def test_main(self):
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

        fa = FieldArray("y", [1.1, 2.2, 3.3, 4.4, 5.5], is_input=True)
        fa.append(10)
        self.assertEqual(fa.pytype, float)
        self.assertEqual(fa.dtype, np.float64)

        fa = FieldArray("y", ["a", "b", "c", "d"], is_input=True)
        fa.append("e")
        self.assertEqual(fa.dtype, np.str)
        self.assertEqual(fa.pytype, str)

    def test_support_np_array(self):
        fa = FieldArray("y", np.array([[1.1, 2.2, 3.3, 4.4, 5.5]]), is_input=True)
        self.assertEqual(fa.dtype, np.float64)
        self.assertEqual(fa.pytype, float)

        fa.append(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
        self.assertEqual(fa.dtype, np.float64)
        self.assertEqual(fa.pytype, float)

        fa = FieldArray("my_field", np.random.rand(3, 5), is_input=True)
        # in this case, pytype is actually a float. We do not care about it.
        self.assertEqual(fa.dtype, np.float64)

    def test_nested_list(self):
        fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1.1, 2.2, 3.3, 4.4, 5.5]], is_input=True)
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


class TestPadder(unittest.TestCase):

    def test01(self):
        """
        测试AutoPadder能否正常工作
        :return:
        """
        from fastNLP.core.fieldarray import AutoPadder
        padder = AutoPadder()
        content = ['This is a str', 'this is another str']
        self.assertListEqual(content, padder(content, None, np.str).tolist())

        content = [1, 2]
        self.assertListEqual(content, padder(content, None, np.int64).tolist())

        content = [[1,2], [3], [4]]
        self.assertListEqual([[1,2], [3, 0], [4, 0]],
                              padder(content, None, np.int64).tolist())

        content = [
                        [[1, 2, 3], [4, 5], [7,8,9,10]],
                        [[1]]
                    ]
        self.assertListEqual(content,
                             padder(content, None, np.int64).tolist())

    def test02(self):
        """
        测试EngChar2DPadder能不能正确使用
        :return:
        """
        from fastNLP.core.fieldarray import EngChar2DPadder
        padder = EngChar2DPadder(pad_length=0)

        contents = [1, 2]
        # 不能是1维
        with self.assertRaises(ValueError):
            padder(contents, None, np.int64)
        contents = [[1, 2]]
        # 不能是2维
        with self.assertRaises(ValueError):
            padder(contents, None, np.int64)
        contents = [[[[1, 2]]]]
        # 不能是3维以上
        with self.assertRaises(ValueError):
            padder(contents, None, np.int64)

        contents = [
                        [[1, 2, 3], [4, 5], [7,8,9,10]],
                        [[1]]
                    ]
        self.assertListEqual([[[1, 2, 3, 0], [4, 5, 0, 0], [7, 8, 9, 10]], [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],
                             padder(contents, None, np.int64).tolist())

        padder = EngChar2DPadder(pad_length=5, pad_val=-100)
        self.assertListEqual(
            [[[1, 2, 3, -100, -100], [4, 5, -100, -100, -100], [7, 8, 9, 10, -100]],
             [[1, -100, -100, -100, -100], [-100, -100, -100, -100, -100], [-100, -100, -100, -100, -100]]],
            padder(contents, None, np.int64).tolist()
        )
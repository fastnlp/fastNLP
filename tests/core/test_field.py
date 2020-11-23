import unittest

import numpy as np
import torch

from fastNLP import FieldArray
from fastNLP.core.field import _get_ele_type_and_dim
from fastNLP import AutoPadder

class TestFieldArrayTyepDimDetect(unittest.TestCase):
    """
    检测FieldArray能否正确识别type与ndim

    """
    def test_case1(self):
        # 1.1 常规类型测试
        for value in [1, True, 1.0, 'abc']:
            type_ = type(value)
            _type, _dim = _get_ele_type_and_dim(cell=value)
            self.assertListEqual([_type, _dim], [type_, 0])
        # 1.2 mix类型报错
        with self.assertRaises(Exception):
            value = [1, 2, 1.0]
            self.assertRaises(_get_ele_type_and_dim(value))
        # 带有numpy的测试
        # 2.1
        value = np.array([1, 2, 3])
        type_ = value.dtype
        dim_ = 1
        self.assertSequenceEqual(_get_ele_type_and_dim(cell=value), [type_, dim_])
        # 2.2
        value = np.array([[1, 2], [3, 4, 5]]) # char  embedding的场景
        self.assertSequenceEqual([int, 2], _get_ele_type_and_dim(value))
        # 2.3
        value = np.zeros((3, 4))
        self.assertSequenceEqual([value.dtype, 2], _get_ele_type_and_dim(value))
        # 2.4 测试错误的dimension
        with self.assertRaises(Exception):
            value = np.array([[1, 2], [3, [1]]])
            _get_ele_type_and_dim(value)
        # 2.5 测试混合类型
        with self.assertRaises(Exception):
            value = np.array([[1, 2], [3.0]])
            _get_ele_type_and_dim(value)

        # 带有tensor的测试
        # 3.1 word embedding的场景
        value = torch.zeros(3, 10)
        self.assertSequenceEqual([value.dtype, 2], _get_ele_type_and_dim(value))
        # 3.2 char embedding/image的场景
        value = torch.zeros(3, 32, 32)
        self.assertSequenceEqual([value.dtype, 3], _get_ele_type_and_dim(value))


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
        # list of array
        fa = FieldArray("x", [np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])], is_input=True)
        self.assertEqual(fa.dtype, np.array([1]).dtype)

    def test_init_v8(self):
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
        fa = FieldArray("x", [1, 2, 3, 4, 5], is_input=True)
        self.assertEqual(fa.dtype, int)

        fa = FieldArray("y", [1.1, 2.2, 3.3, 4.4, 5.5], is_input=True)
        fa.append(10.0)
        self.assertEqual(fa.dtype, float)

        fa = FieldArray("y", ["a", "b", "c", "d"], is_input=True)
        fa.append("e")
        self.assertEqual(fa.dtype, str)

    def test_support_np_array(self):
        fa = FieldArray("y", np.array([[1.1, 2.2, 3.3, 4.4, 5.5]]), is_input=True)
        self.assertEqual(fa.dtype, np.float64)

        fa.append(np.array([1.1, 2.2, 3.3, 4.4, 5.5]))
        self.assertEqual(fa.dtype, np.float64)

        fa = FieldArray("my_field", np.random.rand(3, 5), is_input=True)
        # in this case, pytype is actually a float. We do not care about it.
        self.assertEqual(fa.dtype, np.float64)

    def test_nested_list(self):
        fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1.1, 2.2, 3.3, 4.4, 5.5]], is_input=True)
        self.assertEqual(fa.dtype, float)

    def test_getitem_v1(self):
        fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1.0, 2.0, 3.0, 4.0, 5.0]], is_input=True)
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
            fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1, 2, 3, 4, 5]], is_input=True, use_1st_ins_infer_dim_type=False)
            fa.append(0)

        with self.assertRaises(Exception):
            fa = FieldArray("y", [1.1, 2.2, 3.3, 4.4, 5.5], is_input=True, use_1st_ins_infer_dim_type=False)
            fa.append([1, 2, 3, 4, 5])

        with self.assertRaises(Exception):
            fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1, 2, 3, 4, 5]], is_input=True, use_1st_ins_infer_dim_type=False)
            fa.append([])

        with self.assertRaises(Exception):
            fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1, 2, 3, 4, 5]], is_input=True, use_1st_ins_infer_dim_type=False)
            fa.append(["str", 0, 0, 0, 1.89])

        fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1.0, 2.0, 3.0, 4.0, 5.0]], is_input=True, use_1st_ins_infer_dim_type=False)
        fa.append([1.2, 2.3, 3.4, 4.5, 5.6])
        self.assertEqual(len(fa), 3)
        self.assertEqual(fa[2], [1.2, 2.3, 3.4, 4.5, 5.6])

    def test_ignore_type(self):
        # 测试新添加的参数ignore_type，用来跳过类型检查
        fa = FieldArray("y", [[1.1, 2.2, "jin", {}, "hahah"], [int, 2, "$", 4, 5]], is_input=True, ignore_type=True)
        fa.append([1.2, 2.3, str, 4.5, print])

        fa = FieldArray("y", [(1, "1"), (2, "2"), (3, "3"), (4, "4")], is_target=True, ignore_type=True)


class TestAutoPadder(unittest.TestCase):
    def test00(self):
        padder = AutoPadder()
        # 没有类型时
        contents = [(1, 2), ('str', 'a')]
        padder(contents, None, None, None)

    def test01(self):
        # 测试使用多维的bool, int, str, float的情况
        # str
        padder = AutoPadder()
        content = ['This is a str', 'this is another str']
        self.assertListEqual(content, padder(content, None, str, 0).tolist())

        # 1维int
        content = [[1, 2, 3], [4,], [5, 6, 7, 8]]
        padded_content = [[1, 2, 3, 0], [4, 0, 0, 0], [5, 6, 7, 8]]
        self.assertListEqual(padder(content, None, int, 1).tolist(), padded_content)

        # 二维int
        padded_content = [[[1, 2, 3, 0], [4, 5, 0, 0], [7, 8, 9, 10]], [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]
        content = [
            [[1, 2, 3], [4, 5], [7, 8, 9, 10]],
            [[1]]
        ]
        self.assertListEqual(padder(content, None, int, 2).tolist(), padded_content)

        # 3维图片
        contents = [np.random.rand(3, 4, 4).tolist() for _ in range(5)]
        self.assertTrue(padder(contents, None, float, 3).shape==(5, 3, 4, 4))

        # 更高维度直接返回
        contents = [np.random.rand(24, 3, 4, 4).tolist() for _ in range(5)]
        self.assertTrue(isinstance(padder(contents, None, float, 4), np.ndarray))

    def test02(self):
        padder = AutoPadder()
        # 测试numpy的情况
        # 0维
        contents = np.arange(12)
        self.assertListEqual(padder(contents, None, contents.dtype, 0).tolist(), contents.tolist())

        # 1维
        contents = np.arange(12).reshape((3, 4))
        self.assertListEqual(padder(contents, None, contents.dtype, 1).tolist(), contents.tolist())

        # 2维
        contents = np.ones((3, 10, 5))
        self.assertListEqual(padder(contents, None, contents.dtype, 2).tolist(), contents.tolist())

        # 3维
        contents = [np.random.rand(3, 4, 4) for _ in range(5)]
        l_contents = [content.tolist() for content in contents]
        self.assertListEqual(padder(contents, None, contents[0].dtype, 3).tolist(), l_contents)

    def test03(self):
        padder = AutoPadder()
        # 测试tensor的情况
        # 0维
        contents = torch.arange(12)
        r_contents = padder(contents, None, contents.dtype, 0)
        self.assertSequenceEqual(r_contents.tolist(), contents.tolist())
        self.assertTrue(r_contents.dtype==contents.dtype)

        # 0维
        contents = [torch.tensor(1) for _ in range(10)]
        self.assertSequenceEqual(padder(contents, None, torch.int64, 0).tolist(), contents)

        # 1维
        contents = torch.randn(3, 4)
        padder(contents, None, torch.float64, 1)

        # 3维
        contents = [torch.randn(3, 4, 4) for _ in range(5)]
        padder(contents, None, torch.float64, 3)



class TestEngChar2DPadder(unittest.TestCase):
    def test01(self):
        """
        测试EngChar2DPadder能不能正确使用
        :return:
        """
        from fastNLP import EngChar2DPadder
        padder = EngChar2DPadder(pad_length=0)

        contents = [1, 2]
        # 不能是0维
        with self.assertRaises(Exception):
            padder(contents, None, np.int64, 0)
        contents = [[1, 2]]
        # 不能是1维
        with self.assertRaises(Exception):
            padder(contents, None, np.int64, 1)
        contents = [
                    [[[[1, 2]]]]
                   ]
        # 不能是3维以上
        with self.assertRaises(Exception):
            padder(contents, None, np.int64, 3)

        contents = [
                        [[1, 2, 3], [4, 5], [7,8,9,10]],
                        [[1]]
                    ]
        self.assertListEqual([[[1, 2, 3, 0], [4, 5, 0, 0], [7, 8, 9, 10]], [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],
                             padder(contents, None, np.int64, 2).tolist())

        padder = EngChar2DPadder(pad_length=5, pad_val=-100)
        self.assertListEqual(
            [[[1, 2, 3, -100, -100], [4, 5, -100, -100, -100], [7, 8, 9, 10, -100]],
             [[1, -100, -100, -100, -100], [-100, -100, -100, -100, -100], [-100, -100, -100, -100, -100]]],
            padder(contents, None, np.int64, 2).tolist()
        )


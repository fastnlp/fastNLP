import os
import sys
import unittest

from fastNLP import DataSet
from fastNLP.core.dataset import ApplyResultException
from fastNLP import FieldArray
from fastNLP import Instance
from fastNLP.io import CSVLoader


class TestDataSetInit(unittest.TestCase):
    """初始化DataSet的办法有以下几种：
    1) 用dict:
        1.1) 二维list  DataSet({"x": [[1, 2], [3, 4]]})
        1.2) 二维array  DataSet({"x": np.array([[1, 2], [3, 4]])})
        1.3) 三维list  DataSet({"x": [[[1, 2], [3, 4]], [[1, 2], [3, 4]]]})
    2) 用list of Instance:
        2.1) 一维list DataSet([Instance(x=[1, 2, 3, 4])])
        2.2) 一维array DataSet([Instance(x=np.array([1, 2, 3, 4]))])
        2.3) 二维list  DataSet([Instance(x=[[1, 2], [3, 4]])])
        2.4) 二维array  DataSet([Instance(x=np.array([[1, 2], [3, 4]]))])

    只接受纯list或者最外层ndarray
    """
    def test_init_v1(self):
        # 一维list
        ds = DataSet([Instance(x=[1, 2, 3, 4], y=[5, 6])] * 40)
        self.assertTrue("x" in ds.field_arrays and "y" in ds.field_arrays)
        self.assertEqual(ds.field_arrays["x"].content, [[1, 2, 3, 4], ] * 40)
        self.assertEqual(ds.field_arrays["y"].content, [[5, 6], ] * 40)

    def test_init_v2(self):
        # 用dict
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


class TestDataSetMethods(unittest.TestCase):
    def test_append(self):
        dd = DataSet()
        for _ in range(3):
            dd.append(Instance(x=[1, 2, 3, 4], y=[5, 6]))
        self.assertEqual(len(dd), 3)
        self.assertEqual(dd.field_arrays["x"].content, [[1, 2, 3, 4]] * 3)
        self.assertEqual(dd.field_arrays["y"].content, [[5, 6]] * 3)

    def test_add_field(self):
        dd = DataSet()
        dd.add_field("x", [[1, 2, 3]] * 10)
        dd.add_field("y", [[1, 2, 3, 4]] * 10)
        dd.add_field("z", [[5, 6]] * 10)
        self.assertEqual(len(dd), 10)
        self.assertEqual(dd.field_arrays["x"].content, [[1, 2, 3]] * 10)
        self.assertEqual(dd.field_arrays["y"].content, [[1, 2, 3, 4]] * 10)
        self.assertEqual(dd.field_arrays["z"].content, [[5, 6]] * 10)

        with self.assertRaises(RuntimeError):
            dd.add_field("??", [[1, 2]] * 40)

    def test_add_field_ignore_type(self):
        dd = DataSet()
        dd.add_field("x", [(1, "1"), (2, "2"), (3, "3"), (4, "4")], ignore_type=True, is_target=True)
        dd.add_field("y", [{1, "1"}, {2, "2"}, {3, "3"}, {4, "4"}], ignore_type=True, is_target=True)

    def test_delete_field(self):
        dd = DataSet()
        dd.add_field("x", [[1, 2, 3]] * 10)
        dd.add_field("y", [[1, 2, 3, 4]] * 10)
        dd.delete_field("x")
        self.assertFalse("x" in dd.field_arrays)
        self.assertTrue("y" in dd.field_arrays)

    def test_delete_instance(self):
        dd = DataSet()
        old_length = 2
        dd.add_field("x", [[1, 2, 3]] * old_length)
        dd.add_field("y", [[1, 2, 3, 4]] * old_length)
        dd.delete_instance(0)
        self.assertEqual(len(dd), old_length-1)
        dd.delete_instance(0)
        self.assertEqual(len(dd), old_length-2)

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

    def test_get_item_error(self):
        with self.assertRaises(RuntimeError):
            ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
            _ = ds[40:]

        with self.assertRaises(KeyError):
            ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
            _ = ds["kom"]

    def test_len_(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        self.assertEqual(len(ds), 40)

        ds = DataSet()
        self.assertEqual(len(ds), 0)

    def test_apply(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        ds.apply(lambda ins: ins["x"][::-1], new_field_name="rx")
        self.assertTrue("rx" in ds.field_arrays)
        self.assertEqual(ds.field_arrays["rx"].content[0], [4, 3, 2, 1])

        ds.apply(lambda ins: len(ins["y"]), new_field_name="y")
        self.assertEqual(ds.field_arrays["y"].content[0], 2)

        res = ds.apply(lambda ins: len(ins["x"]))
        self.assertTrue(isinstance(res, list) and len(res) > 0)
        self.assertTrue(res[0], 4)

        ds.apply(lambda ins: (len(ins["x"]), "hahaha"), new_field_name="k", ignore_type=True)
        # expect no exception raised

    def test_apply_tqdm(self):
        import time
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        def do_nothing(ins):
            time.sleep(0.01)
        ds.apply(do_nothing, use_tqdm=True)
        ds.apply_field(do_nothing, field_name='x', use_tqdm=True)

    def test_apply_cannot_modify_instance(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        def modify_inplace(instance):
            instance['words'] = 1

        with self.assertRaises(TypeError):
            ds.apply(modify_inplace)

    def test_apply_more(self):
    
        T = DataSet({"a": [1, 2, 3], "b": [2, 4, 5]})
        func_1 = lambda x: {"c": x["a"] * 2, "d": x["a"] ** 2}
        func_2 = lambda x: {"c": x * 3, "d": x ** 3}
    
        def func_err_1(x):
            if x["a"] == 1:
                return {"e": x["a"] * 2, "f": x["a"] ** 2}
            else:
                return {"e": x["a"] * 2}
    
        def func_err_2(x):
            if x == 1:
                return {"e": x * 2, "f": x ** 2}
            else:
                return {"e": x * 2}
    
        T.apply_more(func_1)
        self.assertEqual(list(T["c"]), [2, 4, 6])
        self.assertEqual(list(T["d"]), [1, 4, 9])
    
        res = T.apply_field_more(func_2, "a", modify_fields=False)
        self.assertEqual(list(T["c"]), [2, 4, 6])
        self.assertEqual(list(T["d"]), [1, 4, 9])
        self.assertEqual(list(res["c"]), [3, 6, 9])
        self.assertEqual(list(res["d"]), [1, 8, 27])
    
        with self.assertRaises(ApplyResultException) as e:
            T.apply_more(func_err_1)
            print(e)
    
        with self.assertRaises(ApplyResultException) as e:
            T.apply_field_more(func_err_2, "a")
            print(e)

    def test_drop(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6], [7, 8, 9, 0]] * 20})
        ds.drop(lambda ins: len(ins["y"]) < 3, inplace=True)
        self.assertEqual(len(ds), 20)

    def test_contains(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        self.assertTrue("x" in ds)
        self.assertTrue("y" in ds)
        self.assertFalse("z" in ds)

    def test_rename_field(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        ds.rename_field("x", "xx")
        self.assertTrue("xx" in ds)
        self.assertFalse("x" in ds)

        with self.assertRaises(KeyError):
            ds.rename_field("yyy", "oo")

    def test_input_target(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        ds.set_input("x")
        ds.set_target("y")
        self.assertTrue(ds.field_arrays["x"].is_input)
        self.assertTrue(ds.field_arrays["y"].is_target)

        with self.assertRaises(KeyError):
            ds.set_input("xxx")
        with self.assertRaises(KeyError):
            ds.set_input("yyy")

    def test_get_input_name(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        self.assertEqual(ds.get_input_name(), [_ for _ in ds.field_arrays if ds.field_arrays[_].is_input])

    def test_get_target_name(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        self.assertEqual(ds.get_target_name(), [_ for _ in ds.field_arrays if ds.field_arrays[_].is_target])

    def test_split(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        d1, d2 = ds.split(0.1)

    def test_apply2(self):
        def split_sent(ins):
            return ins['raw_sentence'].split()
        csv_loader = CSVLoader(headers=['raw_sentence', 'label'], sep='\t')
        data_bundle = csv_loader.load('tests/data_for_tests/tutorial_sample_dataset.csv')
        dataset = data_bundle.datasets['train']
        dataset.drop(lambda x: len(x['raw_sentence'].split()) == 0, inplace=True)
        dataset.apply(split_sent, new_field_name='words', is_input=True)
        # print(dataset)

    def test_add_field_v2(self):
        ds = DataSet({"x": [3, 4]})
        ds.add_field('y', [['hello', 'world'], ['this', 'is', 'a', 'test']], is_input=True, is_target=True)
        # ds.apply(lambda x:[x['x']]*3, is_input=True, is_target=True, new_field_name='y')
        print(ds)

    def test_save_load(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        ds.save("./my_ds.pkl")
        self.assertTrue(os.path.exists("./my_ds.pkl"))

        ds_1 = DataSet.load("./my_ds.pkl")
        os.remove("my_ds.pkl")

    def test_get_all_fields(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        ans = ds.get_all_fields()
        self.assertEqual(ans["x"].content, [[1, 2, 3, 4]] * 10)
        self.assertEqual(ans["y"].content, [[5, 6]] * 10)

    def test_get_field(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        ans = ds.get_field("x")
        self.assertTrue(isinstance(ans, FieldArray))
        self.assertEqual(ans.content, [[1, 2, 3, 4]] * 10)
        ans = ds.get_field("y")
        self.assertTrue(isinstance(ans, FieldArray))
        self.assertEqual(ans.content, [[5, 6]] * 10)

    def test_add_null(self):
        ds = DataSet()
        with self.assertRaises(RuntimeError) as RE:
            ds.add_field('test', [])

    def test_concat(self):
        """
        测试两个dataset能否正确concat

        """
        ds1 = DataSet({"x": [[1, 2, 3, 4] for i in range(10)], "y": [[5, 6] for i in range(10)]})
        ds2 = DataSet({"x": [[4,3,2,1] for i in range(10)], "y": [[6,5] for i in range(10)]})
        ds3 = ds1.concat(ds2)

        self.assertEqual(len(ds3), 20)

        self.assertListEqual(ds1[9]['x'], [1, 2, 3, 4])
        self.assertListEqual(ds1[10]['x'], [4,3,2,1])

        ds2[0]['x'][0] = 100
        self.assertEqual(ds3[10]['x'][0], 4)  # 不改变copy后的field了

        ds3[10]['x'][0] = -100
        self.assertEqual(ds2[0]['x'][0], 100)  # 不改变copy前的field了

        # 测试inplace
        ds1 = DataSet({"x": [[1, 2, 3, 4] for i in range(10)], "y": [[5, 6] for i in range(10)]})
        ds2 = DataSet({"x": [[4, 3, 2, 1] for i in range(10)], "y": [[6, 5] for i in range(10)]})
        ds3 = ds1.concat(ds2, inplace=True)

        ds2[0]['x'][0] = 100
        self.assertEqual(ds3[10]['x'][0], 4)  # 不改变copy后的field了

        ds3[10]['x'][0] = -100
        self.assertEqual(ds2[0]['x'][0], 100)  # 不改变copy前的field了

        ds3[0]['x'][0] = 100
        self.assertEqual(ds1[0]['x'][0], 100)  # 改变copy前的field了

        # 测试mapping
        ds1 = DataSet({"x": [[1, 2, 3, 4] for i in range(10)], "y": [[5, 6] for i in range(10)]})
        ds2 = DataSet({"X": [[4, 3, 2, 1] for i in range(10)], "Y": [[6, 5] for i in range(10)]})
        ds3 = ds1.concat(ds2, field_mapping={'X':'x', 'Y':'y'})
        self.assertEqual(len(ds3), 20)

        # 测试忽略掉多余的
        ds1 = DataSet({"x": [[1, 2, 3, 4] for i in range(10)], "y": [[5, 6] for i in range(10)]})
        ds2 = DataSet({"X": [[4, 3, 2, 1] for i in range(10)], "Y": [[6, 5] for i in range(10)], 'Z':[0]*10})
        ds3 = ds1.concat(ds2, field_mapping={'X':'x', 'Y':'y'})

        # 测试报错
        ds1 = DataSet({"x": [[1, 2, 3, 4] for i in range(10)], "y": [[5, 6] for i in range(10)]})
        ds2 = DataSet({"X": [[4, 3, 2, 1] for i in range(10)]})
        with self.assertRaises(RuntimeError):
            ds3 = ds1.concat(ds2, field_mapping={'X':'x'})

    def test_no_padder(self):
        ds = DataSet()
        ds.add_field('idx', [1, 2, 3], padder=None)
        self.assertEqual(ds['idx'].padder, None)  # should be None, but AutoPadder

    def test_copy_padder(self):
        from fastNLP.core.field import AutoPadder
        ds = DataSet()
        ds.add_field('idx', [1, 2, 3])
        ds['idx'].set_padder(None)  # workaround of problem 1
        ds.apply_field(lambda x: x, 'idx', 'idx')
        self.assertEqual(ds['idx'].padder, None)  # should be None, but AutoPadder

        ds = DataSet()
        ds.add_field('idx', [1, 2, 3])
        ds.apply_field(lambda x: x, 'idx', 'idx')
        self.assertTrue(isinstance(ds.get_field('idx').padder, AutoPadder))  # should be None, but AutoPadder

class TestDataSetIter(unittest.TestCase):
    def test__repr__(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        for iter in ds:
            self.assertEqual(iter.__repr__(), """+--------------+--------+
| x            | y      |
+--------------+--------+
| [1, 2, 3, 4] | [5, 6] |
+--------------+--------+""")


class TestDataSetFieldMeta(unittest.TestCase):
    def test_print_field_meta(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        ds.print_field_meta()

        ds.set_input('x')
        ds.print_field_meta()

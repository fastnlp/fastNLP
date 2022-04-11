import os
import pytest

import numpy as np

from fastNLP.core.dataset import DataSet, FieldArray, Instance, ApplyResultException


class TestDataSetInit:
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
        assert ("x" in ds.field_arrays and "y" in ds.field_arrays) == True
        assert ds.field_arrays["x"].content == [[1, 2, 3, 4], ] * 40
        assert ds.field_arrays["y"].content == [[5, 6], ] * 40

    def test_init_v2(self):
        # 用dict
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        assert ("x" in ds.field_arrays and "y" in ds.field_arrays) == True
        assert ds.field_arrays["x"].content == [[1, 2, 3, 4], ] * 40
        assert ds.field_arrays["y"].content == [[5, 6], ] * 40

    def test_init_assert(self):
        with pytest.raises(AssertionError):
            _ = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 100})
        with pytest.raises(AssertionError):
            _ = DataSet([[1, 2, 3, 4]] * 10)
        with pytest.raises(ValueError):
            _ = DataSet(0.00001)


class TestDataSetMethods:
    def test_append(self):
        dd = DataSet()
        for _ in range(3):
            dd.append(Instance(x=[1, 2, 3, 4], y=[5, 6]))
        assert len(dd) == 3
        assert dd.field_arrays["x"].content == [[1, 2, 3, 4]] * 3
        assert dd.field_arrays["y"].content == [[5, 6]] * 3

    def test_add_field(self):
        dd = DataSet()
        dd.add_field("x", [[1, 2, 3]] * 10)
        dd.add_field("y", [[1, 2, 3, 4]] * 10)
        dd.add_field("z", [[5, 6]] * 10)
        assert len(dd) == 10
        assert dd.field_arrays["x"].content == [[1, 2, 3]] * 10
        assert dd.field_arrays["y"].content == [[1, 2, 3, 4]] * 10
        assert dd.field_arrays["z"].content == [[5, 6]] * 10

        with pytest.raises(RuntimeError):
            dd.add_field("??", [[1, 2]] * 40)

    def test_delete_field(self):
        dd = DataSet()
        dd.add_field("x", [[1, 2, 3]] * 10)
        dd.add_field("y", [[1, 2, 3, 4]] * 10)
        dd.delete_field("x")
        assert ("x" in dd.field_arrays) == False
        assert "y" in dd.field_arrays

    def test_delete_instance(self):
        dd = DataSet()
        old_length = 2
        dd.add_field("x", [[1, 2, 3]] * old_length)
        dd.add_field("y", [[1, 2, 3, 4]] * old_length)
        dd.delete_instance(0)
        assert len(dd) == old_length - 1
        dd.delete_instance(0)
        assert len(dd) == old_length - 2

    def test_getitem(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        ins_1, ins_0 = ds[0], ds[1]
        assert isinstance(ins_1, Instance) and isinstance(ins_0, Instance) == True
        assert ins_1["x"] == [1, 2, 3, 4]
        assert ins_1["y"] == [5, 6]
        assert ins_0["x"] == [1, 2, 3, 4]
        assert ins_0["y"] == [5, 6]

        sub_ds = ds[:10]
        assert isinstance(sub_ds, DataSet) == True
        assert len(sub_ds) == 10

        sub_ds_1 = ds[[10, 0, 2, 3]]
        assert isinstance(sub_ds_1, DataSet) == True
        assert len(sub_ds_1) == 4

        field_array = ds['x']
        assert isinstance(field_array, FieldArray) == True
        assert len(field_array) == 40

    def test_setitem(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        ds.add_field('i', list(range(len(ds))))
        assert ds.get_field('i').content == list(range(len(ds)))
        import random
        random.shuffle(ds)
        import numpy as np
        np.random.shuffle(ds)
        assert ds.get_field('i').content != list(range(len(ds)))

        ins1 = ds[1]
        ds[2] = ds[1]
        assert ds[2]['x'] == ins1['x'] and ds[2]['y'] == ins1['y']

    def test_get_item_error(self):
        with pytest.raises(RuntimeError):
            ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
            _ = ds[40:]

        with pytest.raises(KeyError):
            ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
            _ = ds["kom"]

    def test_len_(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        assert len(ds) == 40

        ds = DataSet()
        assert len(ds) == 0

    def test_add_fieldarray(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        ds.add_fieldarray('z', FieldArray('z', [[7, 8]] * 40))
        assert ds['z'].content == [[7, 8]] * 40

        with pytest.raises(RuntimeError):
            ds.add_fieldarray('z', FieldArray('z', [[7, 8]] * 10))

        with pytest.raises(TypeError):
            ds.add_fieldarray('z', [1, 2, 4])

    def test_copy_field(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        ds.copy_field('x', 'z')
        assert ds['x'].content == ds['z'].content

    def test_has_field(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        assert ds.has_field('x') == True
        assert ds.has_field('z') == False

    def test_get_field(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        with pytest.raises(KeyError):
            ds.get_field('z')
        x_array = ds.get_field('x')
        assert x_array.content == [[1, 2, 3, 4]] * 40

    def test_get_all_fields(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        field_arrays = ds.get_all_fields()
        assert field_arrays["x"].content == [[1, 2, 3, 4]] * 40
        assert field_arrays['y'].content == [[5, 6]] * 40

    def test_get_field_names(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        field_names = ds.get_field_names()
        assert 'x' in field_names
        assert 'y' in field_names

    def test_apply(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 4000, "y": [[5, 6]] * 4000})
        ds.apply(lambda ins: ins["x"][::-1], new_field_name="rx", progress_desc='rx')
        assert ("rx" in ds.field_arrays) == True
        assert ds.field_arrays["rx"].content[0] == [4, 3, 2, 1]

        ds.apply(lambda ins: len(ins["y"]), new_field_name="y", show_progress_bar=False)
        assert ds.field_arrays["y"].content[0] == 2

        res = ds.apply(lambda ins: len(ins["x"]), num_proc=0, progress_desc="len")
        assert (isinstance(res, list) and len(res) > 0) == True
        assert res[0] == 4

        ds.apply(lambda ins: (len(ins["x"]), "hahaha"), new_field_name="k")
        # expect no exception raised

    def test_apply_progress_bar(self):
        import time
        ds = DataSet({"x": [[1, 2, 3, 4]] * 400, "y": [[5, 6]] * 400})

        def do_nothing(ins):
            time.sleep(0.01)

        ds.apply(do_nothing, show_progress_bar=True, num_proc=0)
        ds.apply_field(do_nothing, field_name='x', show_progress_bar=True)

    def test_apply_cannot_modify_instance(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})

        def modify_inplace(instance):
            instance['words'] = 1

        ds.apply(modify_inplace)
        # with self.assertRaises(TypeError):
        #     ds.apply(modify_inplace)

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
        # print(T['c'][0, 1, 2])
        assert list(T["c"].content) == [2, 4, 6]
        assert list(T["d"].content) == [1, 4, 9]

        res = T.apply_field_more(func_2, "a", modify_fields=False)
        assert list(T["c"].content) == [2, 4, 6]
        assert list(T["d"].content) == [1, 4, 9]
        assert list(res["c"]) == [3, 6, 9]
        assert list(res["d"]) == [1, 8, 27]

        with pytest.raises(ApplyResultException) as e:
            T.apply_more(func_err_1)
            print(e)

        with pytest.raises(ApplyResultException) as e:
            T.apply_field_more(func_err_2, "a")
            print(e)

    def test_drop(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6], [7, 8, 9, 0]] * 20})
        ds.drop(lambda ins: len(ins["y"]) < 3, inplace=True)
        assert len(ds) == 20

    def test_contains(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        assert ("x" in ds) == True
        assert ("y" in ds) == True
        assert ("z" in ds) == False

    def test_rename_field(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        ds.rename_field("x", "xx")
        assert ("xx" in ds) == True
        assert ("x" in ds) == False

        with pytest.raises(KeyError):
            ds.rename_field("yyy", "oo")

    def test_split(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        d1, d2 = ds.split(0.1)
        assert len(d2) == (len(ds) * 0.9)
        assert len(d1) == (len(ds) * 0.1)

    def test_add_field_v2(self):
        ds = DataSet({"x": [3, 4]})
        ds.add_field('y', [['hello', 'world'], ['this', 'is', 'a', 'test']])
        # ds.apply(lambda x:[x['x']]*3, new_field_name='y')
        print(ds)

    def test_save_load(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 10, "y": [[5, 6]] * 10})
        ds.save("./my_ds.pkl")
        assert os.path.exists("./my_ds.pkl") == True

        ds_1 = DataSet.load("./my_ds.pkl")
        os.remove("my_ds.pkl")

    def test_add_null(self):
        ds = DataSet()
        with pytest.raises(RuntimeError) as RE:
            ds.add_field('test', [])

    def test_concat(self):
        """
        测试两个dataset能否正确concat

        """
        ds1 = DataSet({"x": [[1, 2, 3, 4] for _ in range(10)], "y": [[5, 6] for _ in range(10)]})
        ds2 = DataSet({"x": [[4, 3, 2, 1] for _ in range(10)], "y": [[6, 5] for _ in range(10)]})
        ds3 = ds1.concat(ds2)

        assert len(ds3) == 20

        assert ds1[9]['x'] == [1, 2, 3, 4]
        assert ds1[10]['x'] == [4, 3, 2, 1]

        ds2[0]['x'][0] = 100
        assert ds3[10]['x'][0] == 4  # 不改变copy后的field了

        ds3[10]['x'][0] = -100
        assert ds2[0]['x'][0] == 100  # 不改变copy前的field了

        # 测试inplace
        ds1 = DataSet({"x": [[1, 2, 3, 4] for i in range(10)], "y": [[5, 6] for i in range(10)]})
        ds2 = DataSet({"x": [[4, 3, 2, 1] for i in range(10)], "y": [[6, 5] for i in range(10)]})
        ds3 = ds1.concat(ds2, inplace=True)

        ds2[0]['x'][0] = 100
        assert ds3[10]['x'][0] == 4  # 不改变copy后的field了

        ds3[10]['x'][0] = -100
        assert ds2[0]['x'][0] == 100  # 不改变copy前的field了

        ds3[0]['x'][0] = 100
        assert ds1[0]['x'][0] == 100  # 改变copy前的field了

        # 测试mapping
        ds1 = DataSet({"x": [[1, 2, 3, 4] for i in range(10)], "y": [[5, 6] for i in range(10)]})
        ds2 = DataSet({"X": [[4, 3, 2, 1] for i in range(10)], "Y": [[6, 5] for i in range(10)]})
        ds3 = ds1.concat(ds2, field_mapping={'X': 'x', 'Y': 'y'})
        assert len(ds3) == 20

        # 测试忽略掉多余的
        ds1 = DataSet({"x": [[1, 2, 3, 4] for i in range(10)], "y": [[5, 6] for i in range(10)]})
        ds2 = DataSet({"X": [[4, 3, 2, 1] for i in range(10)], "Y": [[6, 5] for i in range(10)], 'Z': [0] * 10})
        ds3 = ds1.concat(ds2, field_mapping={'X': 'x', 'Y': 'y'})

        # 测试报错
        ds1 = DataSet({"x": [[1, 2, 3, 4] for i in range(10)], "y": [[5, 6] for i in range(10)]})
        ds2 = DataSet({"X": [[4, 3, 2, 1] for i in range(10)]})
        with pytest.raises(RuntimeError):
            ds3 = ds1.concat(ds2, field_mapping={'X': 'x'})

    def test_instance_field_disappear_bug(self):
        data = DataSet({'raw_chars': [[0, 1], [2]], 'target': [0, 1]})
        data.copy_field(field_name='raw_chars', new_field_name='chars')
        _data = data[:1]
        for field_name in ['raw_chars', 'target', 'chars']:
            assert _data.has_field(field_name) == True

    def test_from_pandas(self):
        import pandas as pd

        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        ds = DataSet.from_pandas(df)
        print(ds)
        assert ds['x'].content == [1, 2, 3]
        assert ds['y'].content == [4, 5, 6]

    def test_to_pandas(self):
        ds = DataSet({'x': [1, 2, 3], 'y': [4, 5, 6]})
        df = ds.to_pandas()

    def test_to_csv(self):
        ds = DataSet({'x': [1, 2, 3], 'y': [4, 5, 6]})
        ds.to_csv("1.csv")
        assert os.path.exists("1.csv") == True
        os.remove("1.csv")

    def test_add_collate_fn(self):
        ds = DataSet({'x': [1, 2, 3], 'y': [4, 5, 6]})

        def collate_fn(item):
            return item

        ds.add_collate_fn(collate_fn)

    def test_get_collator(self):
        from typing import Callable
        ds = DataSet({'x': [1, 2, 3], 'y': [4, 5, 6]})
        collate_fn = ds.get_collator()
        assert isinstance(collate_fn, Callable) == True

    def test_add_seq_len(self):
        ds = DataSet({'x': [[1, 2], [2, 3, 4], [3]], 'y': [4, 5, 6]})
        ds.add_seq_len('x')
        print(ds)

    def test_set_target(self):
        ds = DataSet({'x': [[1, 2], [2, 3, 4], [3]], 'y': [4, 5, 6]})
        ds.set_target('x')


class TestFieldArrayInit:
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
        fa = FieldArray("x", [[1, 2], [3, 4]] * 5)

    def test_init_v2(self):
        # 二维array
        fa = FieldArray("x", np.array([[1, 2], [3, 4]] * 5))

    def test_init_v3(self):
        # 三维list
        fa = FieldArray("x", [[[1, 2], [3, 4]], [[1, 2], [3, 4]]])

    def test_init_v4(self):
        # 一维list
        val = [1, 2, 3, 4]
        fa = FieldArray("x", [val])
        fa.append(val)

    def test_init_v5(self):
        # 一维array
        val = np.array([1, 2, 3, 4])
        fa = FieldArray("x", [val])
        fa.append(val)

    def test_init_v6(self):
        # 二维array
        val = [[1, 2], [3, 4]]
        fa = FieldArray("x", [val])
        fa.append(val)

    def test_init_v7(self):
        # list of array
        fa = FieldArray("x", [np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]])])

    def test_init_v8(self):
        # 二维list
        val = np.array([[1, 2], [3, 4]])
        fa = FieldArray("x", [val])
        fa.append(val)


class TestFieldArray:
    def test_main(self):
        fa = FieldArray("x", [1, 2, 3, 4, 5])
        assert len(fa) == 5
        fa.append(6)
        assert len(fa) == 6

        assert fa[-1] == 6
        assert fa[0] == 1
        fa[-1] = 60
        assert fa[-1] == 60

        assert fa.get(0) == 1
        assert isinstance(fa.get([0, 1, 2]), np.ndarray) == True
        assert list(fa.get([0, 1, 2])) == [1, 2, 3]

    def test_getitem_v1(self):
        fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1.0, 2.0, 3.0, 4.0, 5.0]])
        assert fa[0] == [1.1, 2.2, 3.3, 4.4, 5.5]
        ans = fa[[0, 1]]
        assert isinstance(ans, np.ndarray) == True
        assert isinstance(ans[0], np.ndarray) == True
        assert ans[0].tolist() == [1.1, 2.2, 3.3, 4.4, 5.5]
        assert ans[1].tolist() == [1, 2, 3, 4, 5]
        assert ans.dtype == np.float64

    def test_getitem_v2(self):
        x = np.random.rand(10, 5)
        fa = FieldArray("my_field", x)
        indices = [0, 1, 3, 4, 6]
        for a, b in zip(fa[indices], x[indices]):
            assert a.tolist() == b.tolist()

    def test_append(self):
        fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1.0, 2.0, 3.0, 4.0, 5.0]])
        fa.append([1.2, 2.3, 3.4, 4.5, 5.6])
        assert len(fa) == 3
        assert fa[2] == [1.2, 2.3, 3.4, 4.5, 5.6]

    def test_pop(self):
        fa = FieldArray("y", [[1.1, 2.2, 3.3, 4.4, 5.5], [1.0, 2.0, 3.0, 4.0, 5.0]])
        fa.pop(0)
        assert len(fa) == 1
        assert fa[0] == [1.0, 2.0, 3.0, 4.0, 5.0]
        fa[0] = [1.1, 2.2, 3.3, 4.4, 5.5]
        assert fa[0] == [1.1, 2.2, 3.3, 4.4, 5.5]


class TestCase:

    def test_init(self):
        fields = {"x": [1, 2, 3], "y": [4, 5, 6]}
        ins = Instance(x=[1, 2, 3], y=[4, 5, 6])
        assert isinstance(ins.fields, dict) == True
        assert ins.fields == fields

        ins = Instance(**fields)
        assert ins.fields == fields

    def test_add_field(self):
        fields = {"x": [1, 2, 3], "y": [4, 5, 6]}
        ins = Instance(**fields)
        ins.add_field("z", [1, 1, 1])
        fields.update({"z": [1, 1, 1]})
        assert ins.fields == fields

    def test_get_item(self):
        fields = {"x": [1, 2, 3], "y": [4, 5, 6], "z": [1, 1, 1]}
        ins = Instance(**fields)
        assert ins["x"] == [1, 2, 3]
        assert ins["y"] == [4, 5, 6]
        assert ins["z"] == [1, 1, 1]

    def test_repr(self):
        fields = {"x": [1, 2, 3], "y": [4, 5, 6], "z": [1, 1, 1]}
        ins = Instance(**fields)
        # simple print, that is enough.
        print(ins)

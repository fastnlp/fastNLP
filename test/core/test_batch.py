import unittest

import numpy as np
import torch

from fastNLP import DataSetIter, TorchLoaderIter
from fastNLP import DataSet
from fastNLP import Instance
from fastNLP import SequentialSampler, ConstantTokenNumSampler
from fastNLP import ConcatCollateFn


def generate_fake_dataset(num_samples=1000):
    """
    产生的DataSet包含以下的field {'1':[], '2':[], '3': [], '4':[]}
    :param num_samples: sample的数量
    :return:
    """
    
    max_len = 50
    min_len = 10
    num_features = 4
    
    data_dict = {}
    for i in range(num_features):
        data = []
        lengths = np.random.randint(min_len, max_len, size=(num_samples))
        for length in lengths:
            data.append(np.random.randint(1, 100, size=length))
        data_dict[str(i)] = data
    
    dataset = DataSet(data_dict)
    
    for i in range(num_features):
        if np.random.randint(2) == 0:
            dataset.set_input(str(i))
        else:
            dataset.set_target(str(i))
    return dataset


def construct_dataset(sentences):
    """Construct a data set from a list of sentences.

    :param sentences: list of list of str
    :return dataset: a DataSet object
    """
    dataset = DataSet()
    for sentence in sentences:
        instance = Instance()
        instance['raw_sentence'] = sentence
        dataset.append(instance)
    return dataset


class TestCase1(unittest.TestCase):
    def test_simple(self):
        dataset = construct_dataset(
            [["FastNLP", "is", "the", "most", "beautiful", "tool", "in", "the", "world"] for _ in range(40)])
        dataset.set_target()
        batch = DataSetIter(dataset, batch_size=4, sampler=SequentialSampler(), as_numpy=True)
        
        cnt = 0
        for _, _ in batch:
            cnt += 1
        self.assertEqual(cnt, 10)

    def test_dataset_batching(self):
        ds = DataSet({"x": [[1, 2, 3, 4]] * 40, "y": [[5, 6]] * 40})
        ds.set_input("x")
        ds.set_target("y")
        iter = DataSetIter(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=True)
        for x, y in iter:
            self.assertTrue(isinstance(x["x"], np.ndarray) and isinstance(y["y"], np.ndarray))
            self.assertEqual(len(x["x"]), 4)
            self.assertEqual(len(y["y"]), 4)
            self.assertListEqual(list(x["x"][-1]), [1, 2, 3, 4])
            self.assertListEqual(list(y["y"][-1]), [5, 6])
    
    def test_list_padding(self):
        ds = DataSet({"x": [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]] * 10,
                      "y": [[4, 3, 2, 1], [3, 2, 1], [2, 1], [1]] * 10})
        ds.set_input("x")
        ds.set_target("y")
        iter = DataSetIter(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=True)
        for x, y in iter:
            self.assertEqual(x["x"].shape, (4, 4))
            self.assertEqual(y["y"].shape, (4, 4))
    
    def test_numpy_padding(self):
        ds = DataSet({"x": np.array([[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]] * 10),
                      "y": np.array([[4, 3, 2, 1], [3, 2, 1], [2, 1], [1]] * 10)})
        ds.set_input("x")
        ds.set_target("y")
        iter = DataSetIter(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=True)
        for x, y in iter:
            self.assertEqual(x["x"].shape, (4, 4))
            self.assertEqual(y["y"].shape, (4, 4))
    
    def test_list_to_tensor(self):
        ds = DataSet({"x": [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]] * 10,
                      "y": [[4, 3, 2, 1], [3, 2, 1], [2, 1], [1]] * 10})
        ds.set_input("x")
        ds.set_target("y")
        iter = DataSetIter(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=False)
        for x, y in iter:
            self.assertTrue(isinstance(x["x"], torch.Tensor))
            self.assertEqual(tuple(x["x"].shape), (4, 4))
            self.assertTrue(isinstance(y["y"], torch.Tensor))
            self.assertEqual(tuple(y["y"].shape), (4, 4))
    
    def test_numpy_to_tensor(self):
        ds = DataSet({"x": np.array([[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]] * 10),
                      "y": np.array([[4, 3, 2, 1], [3, 2, 1], [2, 1], [1]] * 10)})
        ds.set_input("x")
        ds.set_target("y")
        iter = DataSetIter(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=False)
        for x, y in iter:
            self.assertTrue(isinstance(x["x"], torch.Tensor))
            self.assertEqual(tuple(x["x"].shape), (4, 4))
            self.assertTrue(isinstance(y["y"], torch.Tensor))
            self.assertEqual(tuple(y["y"].shape), (4, 4))
    
    def test_list_of_list_to_tensor(self):
        ds = DataSet([Instance(x=[1, 2], y=[3, 4]) for _ in range(2)] +
                     [Instance(x=[1, 2, 3, 4], y=[3, 4, 5, 6]) for _ in range(2)])
        ds.set_input("x")
        ds.set_target("y")
        iter = DataSetIter(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=False)
        for x, y in iter:
            self.assertTrue(isinstance(x["x"], torch.Tensor))
            self.assertEqual(tuple(x["x"].shape), (4, 4))
            self.assertTrue(isinstance(y["y"], torch.Tensor))
            self.assertEqual(tuple(y["y"].shape), (4, 4))
    
    def test_list_of_numpy_to_tensor(self):
        ds = DataSet([Instance(x=np.array([1, 2]), y=np.array([3, 4])) for _ in range(2)] +
                     [Instance(x=np.array([1, 2, 3, 4]), y=np.array([3, 4, 5, 6])) for _ in range(2)])
        ds.set_input("x")
        ds.set_target("y")
        iter = DataSetIter(ds, batch_size=4, sampler=SequentialSampler(), as_numpy=False)
        for x, y in iter:
            print(x, y)
    
    def test_sequential_batch(self):
        batch_size = 32
        num_samples = 1000
        dataset = generate_fake_dataset(num_samples)
        
        batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler())
        for batch_x, batch_y in batch:
            pass

    def test_udf_padder(self):
        from fastNLP.core.field import Padder
        alphas = list('abcdefghijk')
        class UDFPadder(Padder):
            def __init__(self):
                super().__init__()

            def __call__(self, contents, field_name, field_ele_dtype, dim):
                results = [alphas[:con] for con in contents]
                return results

        batch_size = 32
        num_samples = 1000
        dataset = generate_fake_dataset(num_samples)
        contents = np.random.randint(5, size=(num_samples))
        dataset.add_field('test', contents, is_input=True, padder=UDFPadder(),
                          ignore_type=True)

        batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler())
        for batch_x, batch_y in batch:
            test = batch_x['test']
            indices = batch.cur_batch_indices
            cons = contents[indices]
            for con,t in zip(cons, test):
                self.assertEqual(alphas[:con], t)

    def test_collate_fn(self):
        batch_size = 32
        num_samples = 1000
        dataset = generate_fake_dataset(num_samples)
        dataset.set_input('1','2')
        dataset.set_target('0','3')

        fn = ConcatCollateFn(inputs=['1', '2'], output='12', pad_val=0, max_len=0, is_input=True, is_target=False)
        dataset.add_collate_fn(fn, name='demo')
        batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler(), drop_last=True)
        for batch_x, batch_y in batch:
            for i in range(batch_size):
                # print(i)
                self.assertEqual(batch_x['12'][i].sum(), batch_x['1'][i].sum() + batch_x['2'][i].sum())
        dataset.delete_collate_fn(name='demo')

        # 测试非input的情况
        dataset.set_input('1', '2', flag=False)  #
        fn = ConcatCollateFn(inputs=['1', '2'], output='12', pad_val=0, max_len=0, is_input=True, is_target=False)
        dataset.add_collate_fn(fn, name='demo')
        batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler(), drop_last=True)
        for batch_x, batch_y in batch:
            for i in range(batch_size):
                self.assertTrue('12' in batch_x)
        dataset.delete_collate_fn(name='demo')
        dataset.set_input('1', '2', flag=True)  #

        # 测试覆盖其它field的情况
        fn = ConcatCollateFn(inputs=['1', '2'], output='3', pad_val=0, max_len=0, is_input=True, is_target=True)
        dataset.add_collate_fn(fn, name='demo')
        batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler(), drop_last=True)
        for batch_x, batch_y in batch:
            for i in range(batch_size):
                # print(i)
                self.assertEqual(batch_y['3'][i].sum(), batch_x['1'][i].sum() + batch_x['2'][i].sum())
        dataset.delete_collate_fn(name='demo')

        # 测试非input，target的情况
        dataset.set_input('1', '2', flag=False)
        fn = ConcatCollateFn(inputs=['1', '2'], output='3', pad_val=0, max_len=0, is_input=True, is_target=True)
        dataset.add_collate_fn(fn, name='demo')
        batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler(), drop_last=True)
        for batch_x, batch_y in batch:
            for i in range(batch_size):
                # print(i)
                self.assertTrue('3' in batch_x)
                self.assertTrue('3' in batch_y)
        dataset.delete_collate_fn(name='demo')

        # 测试加入非法fn的请
        with self.assertRaises(AssertionError):
            dataset.add_collate_fn(1)

        # 测试collate_fn返回值只有一个的情况
        def demo_collate_fn(ins_list):
            return {'3':1}
        dataset.add_collate_fn(demo_collate_fn, name='demo')
        with self.assertRaises(BaseException):
            batch = DataSetIter(dataset, batch_size=batch_size, sampler=SequentialSampler(), drop_last=True)
            for batch_x, batch_y in batch:
                pass
        dataset.delete_collate_fn(name='demo')

        # 测试多个collate_fn
        dataset.add_collate_fn(demo_collate_fn, name='demo')
        dataset.add_collate_fn(demo_collate_fn, name='demo')
        # 测试删除
        dataset.delete_collate_fn()
        dataset.delete_collate_fn()
        self.assertTrue(dataset.collater.is_empty())

    def test_demo(self):
        import torch

        data = DataSet({
            'x1': [[0, 1],
                   [2]],
            'x2': [[3],
                   [2, 4, 5]
                   ],
            'y': [0, 1]
        })
        data.set_target('y')

        # 所有的collate_fn函数都接受list[(ind1, instance1), (ind2, instance2), ...]作为输入，其中ind1/ind2是该instance在dataset中
        #   的index，instance1/instance2是这次batch取出来的数据，包含了所有的field.
        def concat_collate_fn(ins_list):
            x1 = [ins['x1'] for ind,ins in ins_list]
            x2 = [ins['x2'] for ind,ins in ins_list]
            xs = []
            for i in range(len(ins_list)):
                xs.append(torch.LongTensor(x1[i] + x2[i]))
            # 需要自行pad并转换为tensor，但不需要移动到gpu
            arr = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
            b_x = {'x': arr}
            b_y = {}
            # 返回值一定是两个dict，第一个dict的值会认为是input，第二个dict的值会认为是target. 若名称与已有input或target重复，则
            #   采用返回值。
            return b_x, b_y

        data.add_collate_fn(concat_collate_fn)

        for batch_x, batch_y in DataSetIter(data, batch_size=2, sampler=SequentialSampler()):
            print("batch_x:", batch_x)
            print("batch_y:", batch_y)
            # batch_x: {'x': tensor([[0, 1, 3, 0],
            #                        [2, 2, 4, 5]])}
            # batch_y: {'y': array([0, 1])}

        # 如果取batch过程含有一些参数，可以通过类来实现
        class ConCollateFn:
            def __init__(self, max_len=3):
                self.max_len = max_len
            def __call__(self, ins_list):
                x1 = [ins['x1'] for ind, ins in ins_list]
                x2 = [ins['x2'] for ind, ins in ins_list]
                xs = []
                for i in range(len(ins_list)):
                    xs.append(torch.LongTensor(x1[i] + x2[i])[:self.max_len])
                arr = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=0)
                b_x = {'x': arr}
                b_y = {}
                return b_x, b_y
        data.delete_collate_fn()  # 删除之前的collate_fn
        data.add_collate_fn(ConCollateFn(max_len=3))
        for batch_x, batch_y in DataSetIter(data, batch_size=2, sampler=SequentialSampler()):
            print("batch_x:", batch_x)
            print("batch_y:", batch_y)
            # batch_x: {'x': tensor([[0, 1, 3],
            #                        [2, 2, 4]])}
            # batch_y: {'y': array([0, 1])}

    def testTensorLoaderIter(self):
        class FakeData:
            def __init__(self, return_dict=True):
                self.x = [[1,2,3], [4,5,6]]
                self.return_dict = return_dict

            def __len__(self):
                return len(self.x)

            def __getitem__(self, i):
                x = self.x[i]
                y = 0
                if self.return_dict:
                    return {'x':x}, {'y':y}
                return x, y

        data1 = FakeData()
        def collact_fn(ins_list):
            xs = [ins[0]['x'] for ins in ins_list]
            ys = [ins[1]['y'] for ins in ins_list]
            return {'x':xs}, {'y':ys}
        dataiter = TorchLoaderIter(data1, collate_fn=collact_fn, batch_size=2)
        for x, y in dataiter:
            print(x, y)

    def test_batch_sampler(self):
        # 测试DataSetIter与TorchLoaderIter的batch_sampler能否正常工作
        # DataSetIter
        ds = generate_fake_dataset(5)
        ds.set_input('1')
        class BatchSampler:
            def __init__(self, dataset):
                self.num_samples = len(dataset)

            def __iter__(self):
                index = 0
                indexes = list(range(self.num_samples))
                np.random.shuffle(indexes)
                start_idx = 0
                while index < self.num_samples:
                    if start_idx == 0:
                        end_index = self.num_samples//2
                    else:
                        end_index = self.num_samples
                    yield indexes[start_idx:end_index]
                    index = end_index
                    start_idx = end_index

            def __len__(self):
                return 2

        batch_sampler = BatchSampler(ds)

        data_iter = DataSetIter(ds, batch_size=10, sampler=batch_sampler, as_numpy=False, num_workers=0,
                                pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                                batch_sampler=batch_sampler)
        num_samples = [len(ds)//2, len(ds)-len(ds)//2]
        for idx, (batch_x, batch_y) in enumerate(data_iter):
            self.assertEqual(num_samples[idx], len(batch_x['1']))

        # TorchLoaderIter
        class FakeData:
            def __init__(self):
                self.x = [[1,2,3], [4,5,6], [1,2]]

            def __len__(self):
                return len(self.x)

            def __getitem__(self, i):
                x = self.x[i]
                y = 0
                return x,y

        def collate_fn(ins_list):
            xs = [ins[0] for ins in ins_list]
            ys = [ins[1] for ins in ins_list]
            return {'x':xs}, {'y':ys}

        ds = FakeData()
        batch_sampler = BatchSampler(ds)
        data_iter = TorchLoaderIter(ds, batch_size=10, sampler=batch_sampler,
                 num_workers=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None, collate_fn=collate_fn,
                 batch_sampler=batch_sampler)
        num_samples = [len(ds)//2, len(ds)-len(ds)//2]
        for idx, (batch_x, batch_y) in enumerate(data_iter):
            self.assertEqual(num_samples[idx], len(batch_x['x']))

    def test_ConstantTokenNumSampler(self):
        num_samples = 100
        ds = generate_fake_dataset(num_samples)
        ds.set_input('1')
        ds.add_seq_len('1', 'seq_len')
        ds.set_input('seq_len')

        # 测试token数量不超过
        batch_sampler = ConstantTokenNumSampler(ds.get_field('seq_len'), max_token=120)
        data_iter = DataSetIter(ds, batch_size=10, sampler=batch_sampler, as_numpy=False, num_workers=0,
                                pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                                batch_sampler=batch_sampler)
        sample_count = 0
        for batch_x, batch_y in data_iter:
            self.assertTrue(sum(batch_x['seq_len'])<120)
            sample_count += len(batch_x['seq_len'])
        self.assertEqual(sample_count, num_samples)

        # 测试句子数量不超过
        batch_sampler = ConstantTokenNumSampler(ds.get_field('seq_len'), max_token=120, max_sentence=1)
        data_iter = DataSetIter(ds, batch_size=10, sampler=batch_sampler, as_numpy=False, num_workers=0,
                                pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                                batch_sampler=batch_sampler)
        sample_count = 0
        for batch_x, batch_y in data_iter:
            sample_count += len(batch_x['seq_len'])
            self.assertTrue(sum(batch_x['seq_len'])<120 and len(batch_x['seq_len'])==1)
        self.assertEqual(sample_count, num_samples)

        # 测试need_be_multiple_of
        sample_count = 0
        batch_sampler = ConstantTokenNumSampler(ds.get_field('seq_len'), max_token=120, max_sentence=2, need_be_multiple_of=2)
        data_iter = DataSetIter(ds, batch_size=10, sampler=batch_sampler, as_numpy=False, num_workers=0,
                                pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                                batch_sampler=batch_sampler)
        for batch_x, batch_y in data_iter:
            sample_count += len(batch_x['seq_len'])
            self.assertTrue(sum(batch_x['seq_len'])<120 and len(batch_x['seq_len'])==2)
        self.assertEqual(sample_count, num_samples)

        # 测试token数量不超过, bucket尽量接近
        batch_sampler = ConstantTokenNumSampler(ds.get_field('seq_len'), max_token=120, num_bucket=10)
        data_iter = DataSetIter(ds, batch_size=10, sampler=batch_sampler, as_numpy=False, num_workers=0,
                                pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
                                batch_sampler=batch_sampler)
        sample_count = 0
        for batch_x, batch_y in data_iter:
            sample_count += len(batch_x['seq_len'])
            self.assertTrue(sum(batch_x['seq_len'])<120)
        self.assertEqual(sample_count, num_samples)

    """
    def test_multi_workers_batch(self):
        batch_size = 32
        pause_seconds = 0.01
        num_samples = 1000
        dataset = generate_fake_dataset(num_samples)

        num_workers = 1
        batch = Batch(dataset, batch_size=batch_size, sampler=SequentialSampler(), num_workers=num_workers)
        for batch_x, batch_y in batch:
            time.sleep(pause_seconds)

        num_workers = 2
        batch = Batch(dataset, batch_size=batch_size, sampler=SequentialSampler(), num_workers=num_workers)
        end1 = time.time()
        for batch_x, batch_y in batch:
            time.sleep(pause_seconds)
    """
    """
    def test_pin_memory(self):
        batch_size = 32
        pause_seconds = 0.01
        num_samples = 1000
        dataset = generate_fake_dataset(num_samples)

        batch = Batch(dataset, batch_size=batch_size, sampler=SequentialSampler(), pin_memory=True)
        # 这里发生OOM
        # for batch_x, batch_y in batch:
        #     time.sleep(pause_seconds)

        num_workers = 2
        batch = Batch(dataset, batch_size=batch_size, sampler=SequentialSampler(), num_workers=num_workers,
                      pin_memory=True)
        # 这里发生OOM
        # for batch_x, batch_y in batch:
        #     time.sleep(pause_seconds)
    """

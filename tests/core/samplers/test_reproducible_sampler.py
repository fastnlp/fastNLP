import unittest

from itertools import product
import numpy as np

from functools import partial
from array import array

from fastNLP.core.samplers.reproducible_sampler import RandomSampler, ReproducibleBatchSampler
from fastNLP.core.drivers.torch_driver.utils import replace_batch_sampler
from tests.helpers.datasets.torch_data import TorchNormalDataset



class TestRandomSamplerYh(unittest.TestCase):
    def test_init(self):
        # 测试能否正确初始化
        dataset = TorchNormalDataset(num_of_data=100)
        sampler = RandomSampler(dataset)
        for i in sampler:
            pass

    def test_during_iter(self):
        dataset = TorchNormalDataset(num_of_data=100)
        sampler = RandomSampler(dataset)
        for i in sampler:
            with self.assertRaises(AssertionError):
                sampler.set_distributed(1, 0)
            break

        #  should not raise
        for i in sampler:
            pass
        sampler.set_distributed(1, 0)

    def test_set_distributed(self):
        dataset = TorchNormalDataset(num_of_data=100)
        sampler = RandomSampler(dataset, shuffle=False)
        sampler.set_distributed(num_replicas=2, rank=0, pad=False)
        self.assertEqual(len(sampler), 50)
        count = 0
        for i in sampler:
            self.assertEqual(i%2, 0)
            count += 1
        self.assertEqual(count, 50)

        sampler.set_distributed(num_replicas=2, rank=1, pad=False)
        self.assertEqual(len(sampler), 50)
        count = 0
        for i in sampler:
            self.assertEqual(i%2, 1)
            count += 1
        self.assertEqual(count, 50)

        dataset = TorchNormalDataset(num_of_data=101)
        sampler = RandomSampler(dataset, shuffle=False)
        sampler.set_distributed(num_replicas=2, rank=0, pad=True)
        self.assertEqual(len(sampler), 51)
        count = 0
        for i in sampler:
            self.assertEqual(i%2, 0)
            count += 1
        self.assertEqual(count, 51)

        sampler.set_distributed(num_replicas=2, rank=1, pad=True)
        self.assertEqual(len(sampler), 51)
        count = 0
        for i in sampler:
            if i!=0:
                self.assertEqual(i%2, 1)
            count += 1
        self.assertEqual(count, 51)

    def test_state_dict_check_length(self):
        dataset = TorchNormalDataset(num_of_data=100)
        sampler = RandomSampler(dataset, shuffle=False)
        states = sampler.state_dict()

        new_ds = TorchNormalDataset(num_of_data=10)
        with self.assertRaises(AssertionError):
            new_sampler = RandomSampler(new_ds)
            new_sampler.load_state_dict(states)

        new_ds = TorchNormalDataset(num_of_data=100)
        new_sampler = RandomSampler(new_ds)
        new_sampler.load_state_dict(states)

    def test_state_dict(self):
        num_samples = 100
        dataset = TorchNormalDataset(num_of_data=num_samples)
        # 测试使用 前后shuffle不一致的load操作
        lst = [0]+np.random.randint(1, num_samples, size=3).tolist()
        for pre_shuffle, post_shuffle, num_consumed_samples in product([True, False], [True, False],
                                                                       lst):
            with self.subTest(pre_shuffle=pre_shuffle, post_shuffle=post_shuffle, num_consumed_samples=num_consumed_samples):
                sampler = RandomSampler(dataset, shuffle=pre_shuffle)
                sampler.set_epoch(0)
                already_numbers = set()
                if num_consumed_samples>0:
                    for i, j in enumerate(sampler, start=1):
                        already_numbers.add(j)
                        if i == num_consumed_samples:
                            break
                self.assertEqual(len(already_numbers), num_consumed_samples)

                states = sampler.state_dict()

                new_sampler = RandomSampler(dataset, shuffle=post_shuffle)
                new_sampler.load_state_dict(states)
                new_sampler.set_epoch(0)
                for i in new_sampler:
                    self.assertNotIn(i, already_numbers)

                # 测试切换成多卡也没有问题
                other_rank_number = set()
                for rank in range(3):
                    new_sampler = RandomSampler(dataset, shuffle=post_shuffle)
                    new_sampler.load_state_dict(states)
                    new_sampler.set_distributed(num_replicas=3, rank=rank, pad=False)
                    new_sampler.set_epoch(0)
                    count = 0
                    for i in new_sampler:
                        self.assertNotIn(i, other_rank_number)
                        other_rank_number.add(i)
                        self.assertNotIn(i, already_numbers)
                        count += 1

    def test_state_dict_2(self):
        # 测试一下从多卡切换到单卡，或者切换到不同卡数量的多卡
        num_samples = 100
        dataset = TorchNormalDataset(num_of_data=num_samples)
        # 测试使用 前后shuffle不一致的load操作
        lst = [0]+np.random.randint(1, num_samples//2, size=3).tolist()
        # lst = [30]
        for pre_shuffle, post_shuffle, num_consumed_samples in product([True, False], [True, False],
                                                                       lst):
            with self.subTest(pre_shuffle=pre_shuffle, post_shuffle=post_shuffle, num_consumed_samples=num_consumed_samples):
                already_numbers = set()
                sampler = RandomSampler(dataset, shuffle=pre_shuffle, seed=0)
                sampler.set_distributed(num_replicas=2, rank=0)
                sampler.set_epoch(0)
                if num_consumed_samples>0:
                    for i, j in enumerate(sampler, start=1):
                        already_numbers.add(j)
                        if i == num_consumed_samples:
                            break
                sampler = RandomSampler(dataset, shuffle=pre_shuffle, seed=0)
                sampler.set_epoch(0)
                sampler.set_distributed(num_replicas=2, rank=1)
                if num_consumed_samples>0:
                    for i, j in enumerate(sampler, start=1):
                        already_numbers.add(j)
                        if i == num_consumed_samples:
                            break
                self.assertEqual(len(already_numbers), num_consumed_samples*2)

                states = sampler.state_dict()

                new_sampler = RandomSampler(dataset, shuffle=post_shuffle)
                new_sampler.load_state_dict(states)
                new_sampler.set_epoch(0)
                for i in new_sampler:
                    self.assertNotIn(i, already_numbers)

                # 测试切换成多卡也没有问题
                other_rank_number = set()
                for rank in range(3):
                    new_sampler = RandomSampler(dataset, shuffle=post_shuffle)
                    new_sampler.load_state_dict(states)
                    new_sampler.set_epoch(0)
                    new_sampler.set_distributed(num_replicas=3, rank=rank, pad=False)
                    count = 0
                    for i in new_sampler:
                        self.assertNotIn(i, other_rank_number)
                        other_rank_number.add(i)
                        self.assertNotIn(i, already_numbers)
                        count += 1


class TestRandomSampler(unittest.TestCase):
    # 测试单卡；
    def test_seed_work_when_shuffle_is_true(self):
        data_length = 100

        torch_normal_data = TorchNormalDataset(num_of_data=data_length)
        for shuffle in [True, False]:
            iterable = RandomSampler(dataset=torch_normal_data, shuffle=shuffle)
            # 迭代一些数据，但是不迭代完；
            iterable.set_epoch(1)
            iterator = iter(iterable)
            pre_data = []
            forward_steps = 30
            for _ in range(forward_steps):
                pre_data.append(next(iterator))

            # 看重新生成迭代器是否能够完全重置状态；
            iterator = iter(iterable)
            res = []
            for _ in range(forward_steps):
                res.append(next(iterator))
            assert pre_data == res

    # 测试断点重训；
    # 如果 shuffle，那么下一轮的数据应当与前一轮不一样；并且如果是断点重训，两次的下一轮应当是一样的；
    def test_2(self):
        data_length = 100
        torch_normal_data = TorchNormalDataset(num_of_data=data_length)
        random_sampler_1 = RandomSampler(dataset=torch_normal_data, shuffle=True)

        iterator = iter(random_sampler_1)
        # 第一轮
        random_sampler_1.set_epoch(0)
        first_epoch = []
        forward_steps = 30
        for _ in range(forward_steps):
            first_epoch.append(next(iterator))

        # 先提前保存断点重训的结果;
        state = random_sampler_1.state_dict()

        # 保存第一个 epoch 的之后的结果，用于查看断点重训是否正确；
        first_left_data = []
        while True:
            try:
                first_left_data.append(next(iterator))
            except StopIteration:
                break

        # 第二轮
        random_sampler_1.set_epoch(1)
        iterator = iter(random_sampler_1)
        second_epoch = []
        for _ in range(forward_steps):
            second_epoch.append(next(iterator))

        assert first_epoch != second_epoch

        # 重新加载第一轮的状态，查看断点重训是否正确；
        random_sampler_2 = RandomSampler(dataset=torch_normal_data, shuffle=True)
        random_sampler_2.load_state_dict(state)
        random_sampler_2.set_epoch(0)
        iterator = iter(random_sampler_2)
        re_first_epoch = []
        while True:
            try:
                re_first_epoch.append(next(iterator))
            except StopIteration:
                break
        assert re_first_epoch == first_left_data

        # 查看第二轮的结果是否也是和第一次的第二轮完全一致；
        random_sampler_2.set_epoch(1)
        iterator = iter(random_sampler_2)
        re_second_epoch = []
        for _ in range(forward_steps):
            re_second_epoch.append(next(iterator))
        assert re_second_epoch == second_epoch

    # 多卡；
    # 如果一个 sampler 还没有迭代完，我们又直接 iter(sampler) 那么是否正确（应当生成一个全新的 sampler）？
    def test_3(self):
        data_length = 100

        torch_normal_data = TorchNormalDataset(num_of_data=data_length)
        random_sampler_1 = partial(RandomSampler, dataset=torch_normal_data, shuffle=False)
        random_sampler_2 = partial(RandomSampler, dataset=torch_normal_data, shuffle=True)
        iterable_items = [random_sampler_1, random_sampler_2]

        world_size = 3
        for pad in {True, False}:
            for iterable in iterable_items:
                for rank in range(world_size):
                    each_rank_iterable = iterable()
                    each_rank_iterable.set_epoch(0)
                    each_rank_iterable.set_distributed(num_replicas=world_size, rank=rank, pad=pad)
                    # 迭代一些数据，但是不迭代完；
                    iterator = iter(each_rank_iterable)
                    pre_data = []
                    forward_steps = 10
                    for _ in range(forward_steps):
                        pre_data.append(next(iterator))

                    # 看重新生成迭代器是否能够完全重置状态；
                    iterator = iter(each_rank_iterable)
                    res = []
                    for _ in range(forward_steps):
                        res.append(next(iterator))
                    assert res == pre_data

    # 测试断点重训；
    # 如果 shuffle，那么下一轮的数据应当与前一轮不一样；并且如果是断点重训，两次的下一轮应当是一样的；
    def test_4(self):
        data_length = 100
        torch_normal_data = TorchNormalDataset(num_of_data=data_length)
        random_sampler_1 = partial(RandomSampler, dataset=torch_normal_data, shuffle=True)

        world_size_1 = 2
        forward_steps = 10

        for pad in {True, False}:
            all_rank_state = {}
            all_rank_first_left_data = {}
            all_rank_second_epoch = {}
            for rank in range(world_size_1):
                each_rank_iterable = random_sampler_1()
                each_rank_iterable.set_distributed(num_replicas=world_size_1, rank=rank, pad=pad)
                iterator = iter(each_rank_iterable)
                # 第一轮
                each_rank_iterable.set_epoch(0)
                first_epoch = []
                for _ in range(forward_steps):
                    first_epoch.append(next(iterator))

                # 先提前保存断点重训的结果;
                all_rank_state[rank] = each_rank_iterable.state_dict()

                # 保存第一个 epoch 的之后的结果，用于查看断点重训是否正确；
                first_left_data = []
                while True:
                    try:
                        first_left_data.append(next(iterator))
                    except StopIteration:
                        break
                all_rank_first_left_data[rank] = first_left_data
                # 第二轮
                each_rank_iterable.set_epoch(1)
                iterator = iter(each_rank_iterable)
                second_epoch = []
                for _ in range(forward_steps):
                    second_epoch.append(next(iterator))
                all_rank_second_epoch[rank] = second_epoch
                assert first_epoch != second_epoch

            # 重新加载第一轮的状态，查看断点重训是否正确；
            random_sampler_2 = partial(RandomSampler, dataset=torch_normal_data, shuffle=True)
            for rank in range(world_size_1):
                each_rank_iterable = random_sampler_2()
                each_rank_iterable.set_distributed(num_replicas=world_size_1, rank=rank, pad=pad)
                each_rank_iterable.load_state_dict(all_rank_state[rank])
                each_rank_iterable.set_epoch(0)
                iterator = iter(each_rank_iterable)
                re_first_epoch = []
                while True:
                    try:
                        re_first_epoch.append(next(iterator))
                    except StopIteration:
                        break
                assert re_first_epoch == all_rank_first_left_data[rank]

                # 查看第二轮的结果是否也是和第一次的第二轮完全一致；
                each_rank_iterable.set_epoch(1)
                iterator = iter(each_rank_iterable)
                re_second_epoch = []
                for _ in range(forward_steps):
                    re_second_epoch.append(next(iterator))
                assert re_second_epoch == all_rank_second_epoch[rank]

    # todo 测试 ddp 时 world_size 改变的断点重训；
    def test_5(self):
        ...



class TestReproducibleBatchSampler:
    def test_torch_dataloader_1(self):
        import torch
        from torch.utils.data import DataLoader
        # no shuffle
        before_batch_size = 7
        dataset = TorchNormalDataset(num_of_data=100)
        dataloader = DataLoader(dataset, batch_size=before_batch_size)
        re_batchsampler = ReproducibleBatchSampler(dataloader.batch_sampler, dataloader.batch_size, drop_last=False)
        dataloader = replace_batch_sampler(dataloader, re_batchsampler)

        forward_steps = 3
        iter_dataloader = iter(dataloader)
        for _ in range(forward_steps):
            next(iter_dataloader)

        # 1. 保存状态
        _get_re_batchsampler = dataloader.batch_sampler
        assert isinstance(_get_re_batchsampler, ReproducibleBatchSampler)
        state = _get_re_batchsampler.state_dict()
        assert state == {"index_list": array("I", list(range(100))), "data_idx": forward_steps*before_batch_size,
                         "sampler_type": "ReproducibleBatchSampler"}

        # 2. 断点重训，重新生成一个 dataloader；
        # 不改变 batch_size；
        dataloader = DataLoader(dataset, batch_size=before_batch_size)
        re_batchsampler = ReproducibleBatchSampler(dataloader.batch_sampler, dataloader.batch_size, drop_last=False)
        re_batchsampler.load_state_dict(state)
        dataloader = replace_batch_sampler(dataloader, re_batchsampler)

        real_res = []
        supposed_res = (torch.tensor(list(range(21, 28))), torch.tensor(list(range(28, 35))))
        forward_steps = 2
        iter_dataloader = iter(dataloader)
        for _ in range(forward_steps):
            real_res.append(next(iter_dataloader))

        for i in range(forward_steps):
            assert all(real_res[i] == supposed_res[i])

        # 改变 batch_size；
        after_batch_size = 3
        dataloader = DataLoader(dataset, batch_size=after_batch_size)
        re_batchsampler = ReproducibleBatchSampler(dataloader.batch_sampler, dataloader.batch_size, drop_last=False)
        re_batchsampler.load_state_dict(state)
        dataloader = replace_batch_sampler(dataloader, re_batchsampler)

        real_res = []
        supposed_res = (torch.tensor(list(range(21, 24))), torch.tensor(list(range(24, 27))))
        forward_steps = 2
        iter_dataloader = iter(dataloader)
        for _ in range(forward_steps):
            real_res.append(next(iter_dataloader))

        for i in range(forward_steps):
            assert all(real_res[i] == supposed_res[i])

        # 断点重训的第二轮是否是一个完整的 dataloader；
        # 先把断点重训所在的那一个 epoch 跑完；
        begin_idx = 27
        while True:
            try:
                data = next(iter_dataloader)
                _batch_size = len(data)
                assert all(data == torch.tensor(list(range(begin_idx, begin_idx + _batch_size))))
                begin_idx += _batch_size
            except StopIteration:
                break

        # 开始新的一轮；
        begin_idx = 0
        iter_dataloader = iter(dataloader)
        while True:
            try:
                data = next(iter_dataloader)
                _batch_size = len(data)
                assert all(data == torch.tensor(list(range(begin_idx, begin_idx + _batch_size))))
                begin_idx += _batch_size
            except StopIteration:
                break

    def test_torch_dataloader_2(self):
        # 测试新的一轮的 index list 是重新生成的，而不是沿用上一轮的；
        from torch.utils.data import DataLoader
        # no shuffle
        before_batch_size = 7
        dataset = TorchNormalDataset(num_of_data=100)
        # 开启 shuffle，来检验断点重训后的第二轮的 index list 是不是重新生成的；
        dataloader = DataLoader(dataset, batch_size=before_batch_size, shuffle=True)
        re_batchsampler = ReproducibleBatchSampler(dataloader.batch_sampler, dataloader.batch_size, drop_last=False)
        dataloader = replace_batch_sampler(dataloader, re_batchsampler)

        # 将一轮的所有数据保存下来，看是否恢复的是正确的；
        all_supposed_data = []
        forward_steps = 3
        iter_dataloader = iter(dataloader)
        for _ in range(forward_steps):
            all_supposed_data.extend(next(iter_dataloader).tolist())

        # 1. 保存状态
        _get_re_batchsampler = dataloader.batch_sampler
        assert isinstance(_get_re_batchsampler, ReproducibleBatchSampler)
        state = _get_re_batchsampler.state_dict()

        # 2. 断点重训，重新生成一个 dataloader；
        # 不改变 batch_size；
        dataloader = DataLoader(dataset, batch_size=before_batch_size, shuffle=True)
        re_batchsampler = ReproducibleBatchSampler(dataloader.batch_sampler, dataloader.batch_size, drop_last=False)
        re_batchsampler.load_state_dict(state)
        dataloader = replace_batch_sampler(dataloader, re_batchsampler)

        # 先把这一轮的数据过完；
        pre_index_list = dataloader.batch_sampler.state_dict()["index_list"]
        while True:
            try:
                all_supposed_data.extend(next(iter_dataloader).tolist())
            except StopIteration:
                break
        assert all_supposed_data == list(pre_index_list)

        # 重新开启新的一轮；
        for _ in range(3):
            iter_dataloader = iter(dataloader)
            res = []
            while True:
                try:
                    res.append(next(iter_dataloader))
                except StopIteration:
                    break

    def test_3(self):
        import torch
        from torch.utils.data import DataLoader, RandomSampler, BatchSampler
        before_batch_size = 7
        dataset = TorchNormalDataset(num_of_data=100)
        # 开启 shuffle，来检验断点重训后的第二轮的 index list 是不是重新生成的；
        dataloader = DataLoader(dataset, batch_size=before_batch_size)

        for idx, data in enumerate(dataloader):
            if idx > 3:
                break

        iterator = iter(dataloader)
        for each in iterator:
            pass

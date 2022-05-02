from functools import reduce

from fastNLP.core.controllers.utils.utils import _TruncatedDataLoader  # TODO: 该类修改过，记得将 test 也修改；
from tests.helpers.datasets.normal_data import NormalIterator


class Test_WrapDataLoader:

    def test_normal_generator(self):
        all_sanity_batches = [4, 20, 100]
        for sanity_batches in all_sanity_batches:
            data = NormalIterator(num_of_data=1000)
            wrapper = _TruncatedDataLoader(dataloader=data, num_batches=sanity_batches)
            dataloader = iter(wrapper)
            mark = 0
            while True:
                try:
                    _data = next(dataloader)
                except StopIteration:
                    break
                mark += 1
            assert mark == sanity_batches

    def test_torch_dataloader(self):
        from tests.helpers.datasets.torch_data import TorchNormalDataset
        from torch.utils.data import DataLoader

        bses = [8, 16, 40]
        all_sanity_batches = [4, 7, 10]
        for bs in bses:
            for sanity_batches in all_sanity_batches:
                dataset = TorchNormalDataset(num_of_data=1000)
                dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
                wrapper = _TruncatedDataLoader(dataloader, num_batches=sanity_batches)
                dataloader = iter(wrapper)
                all_supposed_running_data_num = 0
                while True:
                    try:
                        _data = next(dataloader)
                    except StopIteration:
                        break
                    all_supposed_running_data_num += _data.shape[0]
                assert all_supposed_running_data_num == bs * sanity_batches

    def test_len(self):
        from tests.helpers.datasets.torch_data import TorchNormalDataset
        from torch.utils.data import DataLoader

        bses = [8, 16, 40]
        all_sanity_batches = [4, 7, 10]
        length = []
        for bs in bses:
            for sanity_batches in all_sanity_batches:
                dataset = TorchNormalDataset(num_of_data=1000)
                dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
                wrapper = _TruncatedDataLoader(dataloader, num_batches=sanity_batches)
                length.append(len(wrapper))
        assert length == reduce(lambda x, y: x+y, [all_sanity_batches for _ in range(len(bses))])
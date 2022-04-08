from torch.utils.data.sampler import SequentialSampler, RandomSampler

from fastNLP.core.samplers.sampler import ReproduceSampler
from tests.helpers.datasets.normal_data import NormalIterator


class TestReproduceSampler:

    def test_sequentialsampler(self):
        normal_iterator = NormalIterator(num_of_data=20)
        sequential_sampler = SequentialSampler(normal_iterator)

        reproduce_sampler = ReproduceSampler(sequential_sampler)
        # iter_seq_sampler = iter(sequential_sampler)
        # for each in iter_seq_sampler:
        #     print(each)
        iter_reproduce_sampler = iter(reproduce_sampler)
        forward_step = 3
        for _ in range(forward_step):
            next(iter_reproduce_sampler)
        state = reproduce_sampler.save_state()
        assert state["current_batch_idx"] == forward_step

        new_repro_sampler = ReproduceSampler(sequential_sampler)
        assert new_repro_sampler.save_state()["current_batch_idx"] == 0

        new_repro_sampler.load_state(state)
        iter_new_repro_sampler = iter(new_repro_sampler)
        new_index_list = []
        for each in iter_new_repro_sampler:
            new_index_list.append(each)
        assert new_index_list == list(range(3, 20))




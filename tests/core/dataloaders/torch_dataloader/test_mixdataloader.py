import pytest
from typing import Mapping

from fastNLP.core.dataloaders import MixDataLoader
from fastNLP import DataSet
from fastNLP.core.collators import Collator
from fastNLP.envs.imports import _NEED_IMPORT_TORCH

if _NEED_IMPORT_TORCH:
    import torch
    from torch.utils.data import SequentialSampler, RandomSampler

d1 = DataSet({"x": [[1, 2], [2, 3, 4], [4, 5, 6, 7]] * 10, "y": [1, 0, 1] * 10})

d2 = DataSet({'x': [[101, 201], [201, 301, 401], [100]] * 10, 'y': [20, 10, 10] * 10})

d3 = DataSet({'x': [[1000, 2000], [0], [2000, 3000, 4000, 5000]] * 100, 'y': [100, 100, 200] * 100})


def test_pad_val(tensor, val=0):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.tolist()
    for item in tensor:
        if item[-1] > 0:
            continue
        elif item[-1] != val:
            return False
    return True


@pytest.mark.torch
class TestMixDataLoader:

    def test_sequential_init(self):
        datasets = {'d1': d1, 'd2': d2, 'd3': d3}
        # drop_last = True, collate_fn = 'auto
        dl = MixDataLoader(datasets=datasets, mode='sequential', collate_fn='auto', drop_last=True)
        for idx, batch in enumerate(dl):
            if idx == 0:
                # d1
                assert batch['x'].shape == torch.Size([16, 4])
            if idx == 1:
                # d2
                assert batch['x'].shape == torch.Size([16, 3])
            if idx > 1:
                # d3
                assert batch['x'].shape == torch.Size([16, 4])
            assert test_pad_val(batch['x'], val=0)

        # collate_fn = Callable
        def collate_batch(batch):
            new_batch = {'x': [], 'y': []}
            for ins in batch:
                new_batch['x'].append(ins['x'])
                new_batch['y'].append(ins['y'])
            return new_batch

        dl1 = MixDataLoader(datasets=datasets, mode='sequential', collate_fn=collate_batch, drop_last=True)
        for idx, batch in enumerate(dl1):
            if idx == 0:
                # d1
                assert [1, 2] in batch['x']
            if idx == 1:
                # d2
                assert [101, 201] in batch['x']
            if idx > 1:
                # d3
                assert [1000, 2000] in batch['x']
            assert 'x' in batch and 'y' in batch

        collate_fns = {'d1': Collator(backend='auto').set_pad("x", -1),
                       'd2': Collator(backend='auto').set_pad("x", -2),
                       'd3': Collator(backend='auto').set_pad("x", -3)}
        dl2 = MixDataLoader(datasets=datasets, mode='sequential', collate_fn=collate_fns, drop_last=True)
        for idx, batch in enumerate(dl2):
            if idx == 0:
                assert test_pad_val(batch['x'], val=-1)
                assert batch['x'].shape == torch.Size([16, 4])
            if idx == 1:
                assert test_pad_val(batch['x'], val=-2)
                assert batch['x'].shape == torch.Size([16, 3])
            if idx > 1:
                assert test_pad_val(batch['x'], val=-3)
                assert batch['x'].shape == torch.Size([16, 4])

        # sampler 为 str
        dl3 = MixDataLoader(datasets=datasets, mode='sequential', sampler='seq', drop_last=True)
        dl4 = MixDataLoader(datasets=datasets, mode='sequential', sampler='rand', drop_last=True)
        for idx, batch in enumerate(dl3):
            if idx == 0:
                # d1
                assert batch['x'][:3].tolist() == [[1, 2, 0, 0], [2, 3, 4, 0], [4, 5, 6, 7]]
                assert batch['x'].shape == torch.Size([16, 4])
            if idx == 1:
                # d2
                assert batch['x'][:3].tolist() == [[101, 201, 0], [201, 301, 401], [100, 0, 0]]
                assert batch['x'].shape == torch.Size([16, 3])
            if idx == 2:
                # d3
                assert batch['x'][:3].tolist() == [[1000, 2000, 0, 0], [0, 0, 0, 0], [2000, 3000, 4000, 5000]]
            if idx > 1:
                # d3
                assert batch['x'].shape == torch.Size([16, 4])
            assert test_pad_val(batch['x'], val=0)

        for idx, batch in enumerate(dl4):
            if idx == 0:
                # d1
                assert batch['x'][:3].tolist() != [[1, 2, 0, 0], [2, 3, 4, 0], [4, 5, 6, 7]]
                assert batch['x'].shape == torch.Size([16, 4])
            if idx == 1:
                # d2
                assert batch['x'][:3].tolist() != [[101, 201, 0], [201, 301, 401], [100, 0, 0]]
                assert batch['x'].shape == torch.Size([16, 3])
            if idx == 2:
                # d3
                assert batch['x'][:3].tolist() != [[1000, 2000, 0, 0], [0, 0, 0, 0], [2000, 3000, 4000, 5000]]
            if idx > 1:
                # d3
                assert batch['x'].shape == torch.Size([16, 4])
            assert test_pad_val(batch['x'], val=0)

        # sampler 为 Dict
        samplers = {'d1': SequentialSampler(d1),
                    'd2': SequentialSampler(d2),
                    'd3': RandomSampler(d3)}
        dl5 = MixDataLoader(datasets=datasets, mode='sequential', sampler=samplers, drop_last=True)
        for idx, batch in enumerate(dl5):
            if idx == 0:
                # d1
                assert batch['x'][:3].tolist() == [[1, 2, 0, 0], [2, 3, 4, 0], [4, 5, 6, 7]]
                assert batch['x'].shape == torch.Size([16, 4])
            if idx == 1:
                # d2
                assert batch['x'][:3].tolist() == [[101, 201, 0], [201, 301, 401], [100, 0, 0]]
                assert batch['x'].shape == torch.Size([16, 3])
            if idx > 1:
                # d3
                assert batch['x'].shape == torch.Size([16, 4])
            assert test_pad_val(batch['x'], val=0)

        # ds_ratio 为 'truncate_to_least'
        dl6 = MixDataLoader(datasets=datasets, mode='sequential', ds_ratio='truncate_to_least', drop_last=True)
        for idx, batch in enumerate(dl6):
            if idx == 0:
                # d1
                assert batch['x'][:3].tolist() == [[1, 2, 0, 0], [2, 3, 4, 0], [4, 5, 6, 7]]
                assert batch['x'].shape == torch.Size([16, 4])
            if idx == 1:
                # d2
                assert batch['x'][:3].tolist() == [[101, 201, 0], [201, 301, 401], [100, 0, 0]]
                assert batch['x'].shape == torch.Size([16, 3])
            if idx == 2:
                # d3
                assert batch['x'][:3].tolist() == [[1000, 2000, 0, 0], [0, 0, 0, 0], [2000, 3000, 4000, 5000]]
                assert batch['x'].shape == torch.Size([16, 4])
            assert test_pad_val(batch['x'], val=0)
            if idx > 2:
                raise ValueError(f"ds_ratio: 'truncate_to_least' error")

        # ds_ratio 为 'pad_to_most'
        dl7 = MixDataLoader(datasets=datasets, mode='sequential', ds_ratio='pad_to_most', drop_last=True)
        for idx, batch in enumerate(dl7):
            if idx < 18:
                # d1
                assert batch['x'].shape == torch.Size([16, 4])
            if 18 <= idx < 36:
                # d2
                assert batch['x'].shape == torch.Size([16, 3])
            if 36 <= idx < 54:
                # d3
                assert batch['x'].shape == torch.Size([16, 4])
            assert test_pad_val(batch['x'], val=0)
            if idx >= 54:
                raise ValueError(f"ds_ratio: 'pad_to_most' error")

        # ds_ratio 为 Dict[str, float]
        ds_ratio = {'d1': 1.0, 'd2': 2.0, 'd3': 2.0}
        dl8 = MixDataLoader(datasets=datasets, mode='sequential', ds_ratio=ds_ratio, drop_last=True)
        for idx, batch in enumerate(dl8):
            if idx < 1:
                # d1
                assert batch['x'].shape == torch.Size([16, 4])
            if 1 <= idx < 4:
                # d2
                assert batch['x'].shape == torch.Size([16, 3])
            if 4 <= idx < 41:
                # d3
                assert batch['x'].shape == torch.Size([16, 4])
            assert test_pad_val(batch['x'], val=0)
            if idx >= 41:
                raise ValueError(f"ds_ratio: 'pad_to_most' error")

        ds_ratio = {'d1': 0.1, 'd2': 0.6, 'd3': 1.0}
        dl9 = MixDataLoader(datasets=datasets, mode='sequential', ds_ratio=ds_ratio, drop_last=True)
        for idx, batch in enumerate(dl9):
            if idx < 1:
                # d2
                assert batch['x'].shape == torch.Size([16, 3])
            if 1 <= idx < 19:
                # d3
                assert batch['x'].shape == torch.Size([16, 4])

            assert test_pad_val(batch['x'], val=0)
            if idx >= 19:
                raise ValueError(f"ds_ratio: 'pad_to_most' error")

    def test_mix(self):
        datasets = {'d1': d1, 'd2': d2, 'd3': d3}
        dl = MixDataLoader(datasets=datasets, mode='mix', collate_fn='auto', drop_last=True)
        for idx, batch in enumerate(dl):
            assert test_pad_val(batch['x'], val=0)
            if idx >= 22:
                raise ValueError(f"out of range")

        # collate_fn = Callable
        def collate_batch(batch):
            new_batch = {'x': [], 'y': []}
            for ins in batch:
                new_batch['x'].append(ins['x'])
                new_batch['y'].append(ins['y'])
            return new_batch

        dl1 = MixDataLoader(datasets=datasets, mode='mix', collate_fn=collate_batch, drop_last=True)
        for idx, batch in enumerate(dl1):
            assert isinstance(batch['x'], list)
            assert test_pad_val(batch['x'], val=0)
            if idx >= 22:
                raise ValueError(f"out of range")

        collate_fns = {'d1': Collator(backend='auto').set_pad("x", -1),
                       'd2': Collator(backend='auto').set_pad("x", -2),
                       'd3': Collator(backend='auto').set_pad("x", -3)}
        with pytest.raises(ValueError):
            MixDataLoader(datasets=datasets, mode='mix', collate_fn=collate_fns)

        # sampler 为 str
        dl3 = MixDataLoader(datasets=datasets, mode='mix', sampler='seq', drop_last=True)
        for idx, batch in enumerate(dl3):
            assert test_pad_val(batch['x'], val=0)
            if idx >= 22:
                raise ValueError(f"out of range")
        dl4 = MixDataLoader(datasets=datasets, mode='mix', sampler='rand', drop_last=True)
        for idx, batch in enumerate(dl4):
            assert test_pad_val(batch['x'], val=0)
            if idx >= 22:
                raise ValueError(f"out of range")
        # sampler 为 Dict
        samplers = {'d1': SequentialSampler(d1),
                    'd2': SequentialSampler(d2),
                    'd3': RandomSampler(d3)}
        dl5 = MixDataLoader(datasets=datasets, mode='mix', sampler=samplers, drop_last=True)
        for idx, batch in enumerate(dl5):
            assert test_pad_val(batch['x'], val=0)
            if idx >= 22:
                raise ValueError(f"out of range")
        # ds_ratio 为 'truncate_to_least'
        dl6 = MixDataLoader(datasets=datasets, mode='mix', ds_ratio='truncate_to_least')
        d1_len, d2_len, d3_len = 0, 0, 0
        for idx, batch in enumerate(dl6):
            for item in batch['y'].tolist():
                if item in [1, 0, 1]:
                    d1_len += 1
                elif item in [20, 10, 10]:
                    d2_len += 1
                elif item in [100, 100, 200]:
                    d3_len += 1
            if idx >= 6:
                raise ValueError(f"ds_ratio 为 'truncate_to_least'出错了")
        assert d1_len == d2_len == d3_len == 30

        # ds_ratio 为 'pad_to_most'
        dl7 = MixDataLoader(datasets=datasets, mode='mix', ds_ratio='pad_to_most')
        d1_len, d2_len, d3_len = 0, 0, 0
        for idx, batch in enumerate(dl7):
            for item in batch['y'].tolist():
                if item in [1, 0, 1]:
                    d1_len += 1
                elif item in [20, 10, 10]:
                    d2_len += 1
                elif item in [100, 100, 200]:
                    d3_len += 1

            if idx >= 57:
                raise ValueError(f"ds_ratio 为 'pad_to_most'出错了")
        assert d1_len == d2_len == d3_len == 300

        # ds_ratio 为 Dict[str, float]
        ds_ratio = {'d1': 1.0, 'd2': 2.0, 'd3': 2.0}
        dl8 = MixDataLoader(datasets=datasets, mode='mix', ds_ratio=ds_ratio)
        d1_len, d2_len, d3_len = 0, 0, 0
        for idx, batch in enumerate(dl8):
            for item in batch['y'].tolist():
                if item in [1, 0, 1]:
                    d1_len += 1
                elif item in [20, 10, 10]:
                    d2_len += 1
                elif item in [100, 100, 200]:
                    d3_len += 1
            if idx >= 44:
                raise ValueError(f"ds_ratio 为 'Dict'出错了")
        assert d1_len == 30
        assert d2_len == 60
        assert d3_len == 600

        ds_ratio = {'d1': 0.1, 'd2': 0.6, 'd3': 1.0}
        dl9 = MixDataLoader(datasets=datasets, mode='mix', ds_ratio=ds_ratio)
        d1_len, d2_len, d3_len = 0, 0, 0
        for idx, batch in enumerate(dl9):
            for item in batch['y'].tolist():
                if item in [1, 0, 1]:
                    d1_len += 1
                elif item in [20, 10, 10]:
                    d2_len += 1
                elif item in [100, 100, 200]:
                    d3_len += 1
            if idx >= 21:
                raise ValueError(f"ds_ratio 为 'Dict'出错了")

    def test_polling(self):
        datasets = {'d1': d1, 'd2': d2, 'd3': d3}
        dl = MixDataLoader(datasets=datasets, mode='polling', collate_fn='auto', batch_size=18)
        for idx, batch in enumerate(dl):
            if idx == 0 or idx == 3:
                assert batch['x'][:3].tolist() == [[1, 2, 0, 0], [2, 3, 4, 0], [4, 5, 6, 7]]
                assert batch['x'].shape[1] == 4
            elif idx == 1 or idx == 4:
                # d2
                assert batch['x'][:3].tolist() == [[101, 201, 0], [201, 301, 401], [100, 0, 0]]
                assert batch['x'].shape[1] == 3
            elif idx == 2 or 4 < idx <= 20:
                assert batch['x'][:3].tolist() == [[1000, 2000, 0, 0], [0, 0, 0, 0], [2000, 3000, 4000, 5000]]
                assert batch['x'].shape[1] == 4
            if idx > 20:
                raise ValueError(f"out of range")
            test_pad_val(batch['x'], val=0)

        # collate_fn = Callable
        def collate_batch(batch):
            new_batch = {'x': [], 'y': []}
            for ins in batch:
                new_batch['x'].append(ins['x'])
                new_batch['y'].append(ins['y'])
            return new_batch

        dl1 = MixDataLoader(datasets=datasets, mode='polling', collate_fn=collate_batch, batch_size=18)
        for idx, batch in enumerate(dl1):
            if idx == 0 or idx == 3:
                assert batch['x'][:3] == [[1, 2], [2, 3, 4], [4, 5, 6, 7]]
            elif idx == 1 or idx == 4:
                # d2
                assert batch['x'][:3] == [[101, 201], [201, 301, 401], [100]]
            elif idx == 2 or 4 < idx <= 20:
                assert batch['x'][:3] == [[1000, 2000], [0], [2000, 3000, 4000, 5000]]
            if idx > 20:
                raise ValueError(f"out of range")

        collate_fns = {'d1': Collator(backend='auto').set_pad("x", -1),
                       'd2': Collator(backend='auto').set_pad("x", -2),
                       'd3': Collator(backend='auto').set_pad("x", -3)}
        dl1 = MixDataLoader(datasets=datasets, mode='polling', collate_fn=collate_fns, batch_size=18)
        for idx, batch in enumerate(dl1):
            if idx == 0 or idx == 3:
                assert test_pad_val(batch['x'], val=-1)
                assert batch['x'][:3].tolist() == [[1, 2, -1, -1], [2, 3, 4, -1], [4, 5, 6, 7]]
                assert batch['x'].shape[1] == 4
            elif idx == 1 or idx == 4:
                # d2
                assert test_pad_val(batch['x'], val=-2)
                assert batch['x'][:3].tolist() == [[101, 201, -2], [201, 301, 401], [100, -2, -2]]
                assert batch['x'].shape[1] == 3
            elif idx == 2 or 4 < idx <= 20:
                assert test_pad_val(batch['x'], val=-3)
                assert batch['x'][:3].tolist() == [[1000, 2000, -3, -3], [0, -3, -3, -3], [2000, 3000, 4000, 5000]]
                assert batch['x'].shape[1] == 4
            if idx > 20:
                raise ValueError(f"out of range")

        # sampler 为 str
        dl2 = MixDataLoader(datasets=datasets, mode='polling', sampler='seq', batch_size=18)
        dl3 = MixDataLoader(datasets=datasets, mode='polling', sampler='rand', batch_size=18)
        for idx, batch in enumerate(dl2):
            if idx == 0 or idx == 3:
                assert batch['x'][:3].tolist() == [[1, 2, 0, 0], [2, 3, 4, 0], [4, 5, 6, 7]]
                assert batch['x'].shape[1] == 4
            elif idx == 1 or idx == 4:
                # d2
                assert batch['x'][:3].tolist() == [[101, 201, 0], [201, 301, 401], [100, 0, 0]]
                assert batch['x'].shape[1] == 3
            elif idx == 2 or 4 < idx <= 20:
                assert batch['x'][:3].tolist() == [[1000, 2000, 0, 0], [0, 0, 0, 0], [2000, 3000, 4000, 5000]]
                assert batch['x'].shape[1] == 4
            if idx > 20:
                raise ValueError(f"out of range")
            test_pad_val(batch['x'], val=0)
        for idx, batch in enumerate(dl3):
            if idx == 0 or idx == 3:
                assert batch['x'].shape[1] == 4
            elif idx == 1 or idx == 4:
                # d2
                assert batch['x'].shape[1] == 3
            elif idx == 2 or 4 < idx <= 20:
                assert batch['x'].shape[1] == 4
            if idx > 20:
                raise ValueError(f"out of range")
            test_pad_val(batch['x'], val=0)
        # sampler 为 Dict
        samplers = {'d1': SequentialSampler(d1),
                    'd2': SequentialSampler(d2),
                    'd3': RandomSampler(d3)}
        dl4 = MixDataLoader(datasets=datasets, mode='polling', sampler=samplers, batch_size=18)
        for idx, batch in enumerate(dl4):
            if idx == 0 or idx == 3:
                assert batch['x'][:3].tolist() == [[1, 2, 0, 0], [2, 3, 4, 0], [4, 5, 6, 7]]
                assert batch['x'].shape[1] == 4
            elif idx == 1 or idx == 4:
                # d2
                assert batch['x'][:3].tolist() == [[101, 201, 0], [201, 301, 401], [100, 0, 0]]
                assert batch['x'].shape[1] == 3
            elif idx == 2 or 4 < idx <= 20:
                assert batch['x'].shape[1] == 4
            if idx > 20:
                raise ValueError(f"out of range")
            test_pad_val(batch['x'], val=0)

        # ds_ratio 为 'truncate_to_least'
        dl5 = MixDataLoader(datasets=datasets, mode='polling', ds_ratio='truncate_to_least', batch_size=18)
        for idx, batch in enumerate(dl5):
            if idx == 0 or idx == 3:
                assert batch['x'][:3].tolist() == [[1, 2, 0, 0], [2, 3, 4, 0], [4, 5, 6, 7]]
                assert batch['x'].shape[1] == 4
            elif idx == 1 or idx == 4:
                # d2
                assert batch['x'][:3].tolist() == [[101, 201, 0], [201, 301, 401], [100, 0, 0]]
                assert batch['x'].shape[1] == 3
            elif idx == 2 or idx == 5:
                assert batch['x'][:3].tolist() == [[1000, 2000, 0, 0], [0, 0, 0, 0], [2000, 3000, 4000, 5000]]
                assert batch['x'].shape[1] == 4
            if idx > 5:
                raise ValueError(f"out of range")
            test_pad_val(batch['x'], val=0)

        # ds_ratio 为 'pad_to_most'
        dl6 = MixDataLoader(datasets=datasets, mode='polling', ds_ratio='pad_to_most', batch_size=18)
        for idx, batch in enumerate(dl6):
            if idx % 3 == 0:
                # d1
                assert batch['x'][:3].tolist() == [[1, 2, 0, 0], [2, 3, 4, 0], [4, 5, 6, 7]]
                assert batch['x'].shape[1] == 4
            if idx % 3 == 1:
                # d2
                assert batch['x'][:3].tolist() == [[101, 201, 0], [201, 301, 401], [100, 0, 0]]
                assert batch['x'].shape[1] == 3
            if idx % 3 == 2:
                # d3
                assert batch['x'][:3].tolist() == [[1000, 2000, 0, 0], [0, 0, 0, 0], [2000, 3000, 4000, 5000]]
                assert batch['x'].shape[1] == 4
            if idx >= 51:
                raise ValueError(f"out of range")
            test_pad_val(batch['x'], val=0)

        # ds_ratio 为 Dict[str, float]
        ds_ratio = {'d1': 1.0, 'd2': 2.0, 'd3': 2.0}
        dl7 = MixDataLoader(datasets=datasets, mode='polling', ds_ratio=ds_ratio, batch_size=18)
        for idx, batch in enumerate(dl7):
            if idx == 0 or idx == 3:
                assert batch['x'][:3].tolist() == [[1, 2, 0, 0], [2, 3, 4, 0], [4, 5, 6, 7]]
                assert batch['x'].shape[1] == 4
            elif idx == 1 or idx == 4 or idx == 6 or idx == 8:
                # d2
                assert batch['x'][:3].tolist() == [[101, 201, 0], [201, 301, 401], [100, 0, 0]]
                assert batch['x'].shape[1] == 3
            elif idx == 2 or idx == 5 or idx == 7 or idx > 8:
                assert batch['x'][:3].tolist() == [[1000, 2000, 0, 0], [0, 0, 0, 0], [2000, 3000, 4000, 5000]]
                assert batch['x'].shape[1] == 4
            if idx > 39:
                raise ValueError(f"out of range")
            test_pad_val(batch['x'], val=0)

        ds_ratio = {'d1': 0.1, 'd2': 0.6, 'd3': 1.0}
        dl8 = MixDataLoader(datasets=datasets, mode='polling', ds_ratio=ds_ratio, batch_size=18)
        for idx, batch in enumerate(dl8):
            if idx == 0:
                assert batch['x'][:3].tolist() == [[1, 2, 0, 0], [2, 3, 4, 0], [4, 5, 6, 7]]
                assert batch['x'].shape[1] == 4
            elif idx == 1:
                # d2
                assert batch['x'][:3].tolist() == [[101, 201, 0], [201, 301, 401], [100, 0, 0]]
                assert batch['x'].shape[1] == 3
            elif idx > 1:
                assert batch['x'][:3].tolist() == [[1000, 2000, 0, 0], [0, 0, 0, 0], [2000, 3000, 4000, 5000]]
                assert batch['x'].shape[1] == 4

            if idx > 18:
                raise ValueError(f"out of range")
            test_pad_val(batch['x'], val=0)
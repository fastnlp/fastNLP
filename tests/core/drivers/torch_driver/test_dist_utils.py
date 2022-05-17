import os
import pytest

import torch
import torch.distributed as dist
import numpy as np

# print(isinstance((1,), tuple))
# exit()

from fastNLP.core.drivers.torch_driver.dist_utils import fastnlp_torch_all_gather, fastnlp_torch_broadcast_object
from tests.helpers.utils import re_run_current_cmd_for_torch, magic_argv_env_context


@pytest.mark.torch
@magic_argv_env_context
def test_fastnlp_torch_all_gather():
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        if 'LOCAL_RANK' not in os.environ and 'RANK' not in os.environ and 'WORLD_SIZE' not in os.environ:
            os.environ['LOCAL_RANK'] = '0'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '2'
        re_run_current_cmd_for_torch(1, output_from_new_proc='all')
        torch.distributed.init_process_group(backend='nccl')
        torch.distributed.barrier()
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        obj = {
            'tensor': torch.full(size=(2,), fill_value=local_rank, dtype=int).cuda(),
            'numpy': np.full(shape=(2, ), fill_value=local_rank),
            'bool': local_rank%2==0,
            'float': local_rank + 0.1,
            'int': local_rank,
            'dict': {
                'rank': local_rank
            },
            'list': [local_rank]*2,
            'str': f'{local_rank}',
            'tensors': [torch.full(size=(2,), fill_value=local_rank, dtype=int).cuda(),
                        torch.full(size=(2,), fill_value=local_rank, dtype=int).cuda()]
        }
        data = fastnlp_torch_all_gather(obj)
        world_size = int(os.environ['WORLD_SIZE'])
        assert len(data) == world_size
        for i in range(world_size):
            assert (data[i]['tensor']==i).sum()==world_size
            assert data[i]['numpy'][0]==i
            assert data[i]['bool']==(i%2==0)
            assert np.allclose(data[i]['float'], i+0.1)
            assert data[i]['int'] == i
            assert data[i]['dict']['rank'] == i
            assert data[i]['list'][0] == i
            assert data[i]['str'] == f'{i}'
            assert data[i]['tensors'][0][0] == i

        for obj in [1, True, 'xxx']:
            data = fastnlp_torch_all_gather(obj)
            assert len(data)==world_size
            assert data[0]==data[1]

    finally:
        dist.destroy_process_group()

@pytest.mark.torch
@magic_argv_env_context
def test_fastnlp_torch_broadcast_object():
    try:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        if 'LOCAL_RANK' not in os.environ and 'RANK' not in os.environ and 'WORLD_SIZE' not in os.environ:
            os.environ['LOCAL_RANK'] = '0'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '2'
        re_run_current_cmd_for_torch(1, output_from_new_proc='all')
        torch.distributed.init_process_group(backend='nccl')
        torch.distributed.barrier()
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        if os.environ['LOCAL_RANK']=="0":
            obj = {
                'tensor': torch.full(size=(2,), fill_value=local_rank, dtype=int).cuda(),
                'numpy': np.full(shape=(2, ), fill_value=local_rank, dtype=int),
                'bool': local_rank%2==0,
                'float': local_rank + 0.1,
                'int': local_rank,
                'dict': {
                    'rank': local_rank
                },
                'list': [local_rank]*2,
                'str': f'{local_rank}',
                'tensors': [torch.full(size=(2,), fill_value=local_rank, dtype=int).cuda(),
                            torch.full(size=(2,), fill_value=local_rank, dtype=int).cuda()]
            }
        else:
            obj = None
        data = fastnlp_torch_broadcast_object(obj, src=0, device=torch.cuda.current_device())
        i = 0
        assert data['tensor'][0]==0
        assert data['numpy'][0]==0
        assert data['bool']==(i%2==0)
        assert np.allclose(data['float'], i+0.1)
        assert data['int'] == i
        assert data['dict']['rank'] == i
        assert data['list'][0] == i
        assert data['str'] == f'{i}'
        assert data['tensors'][0][0] == i

        for obj in [int(os.environ['LOCAL_RANK']), bool(os.environ['LOCAL_RANK']=='1'), os.environ['LOCAL_RANK']]:
            data = fastnlp_torch_broadcast_object(obj, src=0, device=torch.cuda.current_device())
            assert int(data)==0
    finally:
        dist.destroy_process_group()

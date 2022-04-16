import os

from fastNLP.envs.distributed import rank_zero_call, all_rank_call_context
from tests.helpers.utils import re_run_current_cmd_for_torch, Capturing, magic_argv_env_context


@rank_zero_call
def write_something():
    print(os.environ.get('RANK', '0')*5, flush=True)


def write_other_thing():
    print(os.environ.get('RANK', '0')*5, flush=True)


class PaddleTest:
    # @x54-729
    def test_rank_zero_call(self):
        pass

    def test_all_rank_run(self):
        pass


class JittorTest:
    # @x54-729
    def test_rank_zero_call(self):
        pass

    def test_all_rank_run(self):
        pass


class TestTorch:
    @magic_argv_env_context
    def test_rank_zero_call(self):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        if 'LOCAL_RANK' not in os.environ and 'RANK' not in os.environ and 'WORLD_SIZE' not in os.environ:
            os.environ['LOCAL_RANK'] = '0'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '2'
        re_run_current_cmd_for_torch(1, output_from_new_proc='all')
        with Capturing() as output:
            write_something()
        output = output[0]

        if os.environ['LOCAL_RANK'] == '0':
            assert '00000' in output and '11111' not in output
        else:
            assert '00000' not in output and '11111' not in output

        with Capturing() as output:
            rank_zero_call(write_other_thing)()

        output = output[0]
        if os.environ['LOCAL_RANK'] == '0':
            assert '00000' in output and '11111' not in output
        else:
            assert '00000' not in output and '11111' not in output

    @magic_argv_env_context
    def test_all_rank_run(self):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        if 'LOCAL_RANK' not in os.environ and 'RANK' not in os.environ and 'WORLD_SIZE' not in os.environ:
            os.environ['LOCAL_RANK'] = '0'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '2'
        re_run_current_cmd_for_torch(1, output_from_new_proc='all')
        # torch.distributed.init_process_group(backend='nccl')
        # torch.distributed.barrier()
        with all_rank_call_context():
            with Capturing(no_del=True) as output:
                write_something()
        output = output[0]

        if os.environ['LOCAL_RANK'] == '0':
            assert '00000' in output
        else:
            assert '11111' in output

        with all_rank_call_context():
            with Capturing(no_del=True) as output:
                rank_zero_call(write_other_thing)()

        output = output[0]
        if os.environ['LOCAL_RANK'] == '0':
            assert '00000' in output
        else:
            assert '11111' in output
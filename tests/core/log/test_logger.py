import os
import tempfile
import datetime
from pathlib import Path
import logging
import re

from fastNLP.envs.env import FASTNLP_LAUNCH_TIME
from tests.helpers.utils import magic_argv_env_context
from fastNLP.core import synchronize_safe_rm


# 测试 TorchDDPDriver；
@magic_argv_env_context
def test_add_file_ddp_1():
    """
    测试 path 是一个文件的地址，但是这个文件所在的文件夹存在；

    多卡时根据时间创造文件名字有一个很大的 bug，就是不同的进程启动之间是有时差的，因此会导致他们各自输出到单独的 log 文件中；
    """
    import torch
    import torch.distributed as dist

    from fastNLP.core.log.logger import logger
    from fastNLP.core.drivers.torch_driver.ddp import TorchDDPDriver
    from tests.helpers.models.torch_model import TorchNormalModel_Classification_1

    model = TorchNormalModel_Classification_1(num_labels=3, feature_dimension=10)

    driver = TorchDDPDriver(
        model=model,
        parallel_device=[torch.device("cuda:0"), torch.device("cuda:1")],
        output_from_new_proc="all"
    )
    driver.setup()
    msg = 'some test log msg'

    path = Path.cwd()
    filepath = path.joinpath('log.txt')
    handler = logger.add_file(filepath, mode="w")
    logger.info(msg)
    logger.warning(f"\nrank {driver.get_local_rank()} should have this message!\n")

    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            h.flush()
    dist.barrier()
    with open(filepath, 'r') as f:
        line = ''.join([l for l in f])
    assert msg in line
    assert f"\nrank {driver.get_local_rank()} should have this message!\n" in line

    pattern = re.compile(msg)
    assert len(pattern.findall(line)) == 1

    synchronize_safe_rm(filepath)
    dist.barrier()
    dist.destroy_process_group()
    logger.removeHandler(handler)


@magic_argv_env_context
def test_add_file_ddp_2():
    """
    测试 path 是一个文件的地址，但是这个文件所在的文件夹不存在；
    """

    import torch
    import torch.distributed as dist

    from fastNLP.core.log.logger import logger
    from fastNLP.core.drivers.torch_driver.ddp import TorchDDPDriver
    from tests.helpers.models.torch_model import TorchNormalModel_Classification_1

    model = TorchNormalModel_Classification_1(num_labels=3, feature_dimension=10)

    driver = TorchDDPDriver(
        model=model,
        parallel_device=[torch.device("cuda:0"), torch.device("cuda:1")],
        output_from_new_proc="all"
    )
    driver.setup()

    msg = 'some test log msg'

    origin_path = Path.cwd()
    try:
        path = origin_path.joinpath("not_existed")
        filepath = path.joinpath('log.txt')
        handler = logger.add_file(filepath)
        logger.info(msg)
        logger.warning(f"\nrank {driver.get_local_rank()} should have this message!\n")
        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                h.flush()
        dist.barrier()
        with open(filepath, 'r') as f:
            line = ''.join([l for l in f])

        assert msg in line
        assert f"\nrank {driver.get_local_rank()} should have this message!\n" in line
        pattern = re.compile(msg)
        assert len(pattern.findall(line)) == 1
    finally:
        synchronize_safe_rm(path)
        logger.removeHandler(handler)

    dist.barrier()
    dist.destroy_process_group()


@magic_argv_env_context
def test_add_file_ddp_3():
    """
    path = None;

    多卡时根据时间创造文件名字有一个很大的 bug，就是不同的进程启动之间是有时差的，因此会导致他们各自输出到单独的 log 文件中；
    """
    import torch
    import torch.distributed as dist

    from fastNLP.core.log.logger import logger
    from fastNLP.core.drivers.torch_driver.ddp import TorchDDPDriver
    from tests.helpers.models.torch_model import TorchNormalModel_Classification_1

    model = TorchNormalModel_Classification_1(num_labels=3, feature_dimension=10)

    driver = TorchDDPDriver(
        model=model,
        parallel_device=[torch.device("cuda:0"), torch.device("cuda:1")],
        output_from_new_proc="all"
    )
    driver.setup()
    msg = 'some test log msg'

    handler = logger.add_file()
    logger.info(msg)
    logger.warning(f"\nrank {driver.get_local_rank()} should have this message!\n")

    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            h.flush()
    dist.barrier()
    file = Path.cwd().joinpath(os.environ.get(FASTNLP_LAUNCH_TIME)+".log")
    with open(file, 'r') as f:
        line = ''.join([l for l in f])

    # print(f"\nrank: {driver.get_local_rank()} line, {line}\n")
    assert msg in line
    assert f"\nrank {driver.get_local_rank()} should have this message!\n" in line

    pattern = re.compile(msg)
    assert len(pattern.findall(line)) == 1

    synchronize_safe_rm(file)
    dist.barrier()
    dist.destroy_process_group()
    logger.removeHandler(handler)

@magic_argv_env_context
def test_add_file_ddp_4():
    """
    测试 path 是文件夹；
    """

    import torch
    import torch.distributed as dist

    from fastNLP.core.log.logger import logger
    from fastNLP.core.drivers.torch_driver.ddp import TorchDDPDriver
    from tests.helpers.models.torch_model import TorchNormalModel_Classification_1

    model = TorchNormalModel_Classification_1(num_labels=3, feature_dimension=10)

    driver = TorchDDPDriver(
        model=model,
        parallel_device=[torch.device("cuda:0"), torch.device("cuda:1")],
        output_from_new_proc="all"
    )
    driver.setup()
    msg = 'some test log msg'

    path = Path.cwd().joinpath("not_existed")
    try:
        handler = logger.add_file(path)
        logger.info(msg)
        logger.warning(f"\nrank {driver.get_local_rank()} should have this message!\n")

        for h in logger.handlers:
            if isinstance(h, logging.FileHandler):
                h.flush()
        dist.barrier()

        file = path.joinpath(os.environ.get(FASTNLP_LAUNCH_TIME) + ".log")
        with open(file, 'r') as f:
            line = ''.join([l for l in f])
        assert msg in line
        assert f"\nrank {driver.get_local_rank()} should have this message!\n" in line
        pattern = re.compile(msg)
        assert len(pattern.findall(line)) == 1
    finally:
        synchronize_safe_rm(path)
        logger.removeHandler(handler)

    dist.barrier()
    dist.destroy_process_group()


class TestLogger:
    msg = 'some test log msg'

    def test_add_file_1(self):
        """
        测试 path 是一个文件的地址，但是这个文件所在的文件夹存在；
        """
        from fastNLP.core.log.logger import logger

        path = Path(tempfile.mkdtemp())
        try:
            filepath = path.joinpath('log.txt')
            handler = logger.add_file(filepath)
            logger.info(self.msg)
            with open(filepath, 'r') as f:
                line = ''.join([l for l in f])
            assert self.msg in line
        finally:
            synchronize_safe_rm(path)
            logger.removeHandler(handler)

    def test_add_file_2(self):
        """
        测试 path 是一个文件的地址，但是这个文件所在的文件夹不存在；
        """
        from fastNLP.core.log.logger import logger

        origin_path = Path(tempfile.mkdtemp())

        try:
            path = origin_path.joinpath("not_existed")
            path = path.joinpath('log.txt')
            handler = logger.add_file(path)
            logger.info(self.msg)
            with open(path, 'r') as f:
                line = ''.join([l for l in f])
            assert self.msg in line
        finally:
            synchronize_safe_rm(origin_path)
            logger.removeHandler(handler)

    def test_add_file_3(self):
        """
        测试 path 是 None；
        """
        from fastNLP.core.log.logger import logger

        handler = logger.add_file()
        logger.info(self.msg)

        path = Path.cwd()
        cur_datetime = str(datetime.datetime.now().strftime('%Y-%m-%d'))
        for file in path.iterdir():
            if file.name.startswith(cur_datetime):
                with open(file, 'r') as f:
                    line = ''.join([l for l in f])
                assert self.msg in line
                file.unlink()
        logger.removeHandler(handler)

    def test_add_file_4(self):
        """
        测试 path 是文件夹；
        """
        from fastNLP.core.log.logger import logger

        path = Path(tempfile.mkdtemp())
        try:
            handler = logger.add_file(path)
            logger.info(self.msg)

            cur_datetime = str(datetime.datetime.now().strftime('%Y-%m-%d'))
            for file in path.iterdir():
                if file.name.startswith(cur_datetime):
                    with open(file, 'r') as f:
                        line = ''.join([l for l in f])
                    assert self.msg in line
        finally:
            synchronize_safe_rm(path)
            logger.removeHandler(handler)

    def test_stdout(self, capsys):
        from fastNLP.core.log.logger import logger

        handler = logger.set_stdout(stdout="raw")
        logger.info(self.msg)
        logger.debug('aabbc')
        captured = capsys.readouterr()
        assert "some test log msg\n" == captured.out

        logger.removeHandler(handler)


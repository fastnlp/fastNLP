import os
import sys
import __main__
import tempfile
import copy
from typing import List

from fastNLP.core.drivers.utils import distributed_open_proc
from fastNLP.envs.env import (
    FASTNLP_DISTRIBUTED_CHECK,
    FASTNLP_LOG_LEVEL,
    FASTNLP_GLOBAL_SEED,
    USER_CUDA_VISIBLE_DEVICES,
)
from .utils import (
    find_free_ports,
    reset_seed,
)

# 记录各个进程信息
class SubTrainer(object):
    """
    用于统计节点内不同训练进程的信息，和 fastnlp 的 Triainer 没有关系
    """
    def __init__(self, endpoint=None, rank=None):
        self.devices = []
        self.endpoint = endpoint
        self.rank = rank


class FleetLauncher:
    """
    复原了 paddle 的 launch_collective 函数，将其简化后集成到一个类里
    仅支持每个机器单卡的情况。
    """
    def __init__(
        self,
        devices: List[int],
        output_from_new_proc: str = "only_error"
    ):

        self.devices = devices
        self.output_from_new_proc = output_from_new_proc

        self.setup()

    def setup(self):
        """
        进行初始化设置的函数，根据传入的设备找到分布式训练使用的端口号
        """
        self.set_endpoints()
        self.sub_trainers = self.get_process_info()

    def launch(self):
        """
        用于启动分布式进程。
        首先设置 PaddlePaddle 分布式训练需要设置的环境变量，然后建立新的子进程
        """
        # 设置环境变量
        self.global_envs = self.get_global_env()
        self.open_subprocess()
        reset_seed()

    def open_subprocess(self):
        """
        从 sub_trainers 中获取各个 rank 的信息，并且使用 subprocess.Popen 建立新的子进程。
        """

        if __main__.__spec__ is None:
            # Script called as `python a/b/c.py`
            # when user is using hydra find the absolute path
            path_lib = os.path.abspath

            # pull out the commands used to run the script and resolve the abs file path
            command = sys.argv
            try:
                full_path = path_lib(command[0])
            except Exception:
                full_path = os.path.abspath(command[0])

            command[0] = full_path
            # use the same python interpreter and actually running
            command = [sys.executable] + command
        else:  # Script called as `python -m a.b.c`
            command = [sys.executable, "-m", __main__.__spec__._name] + sys.argv[1:]

        current_env = copy.copy(self.global_envs)
        for idx, t in enumerate(self.sub_trainers):
            # 根据不同的 rank 设置环境变量
            proc_env = {
                # global_rank
                "PADDLE_TRAINER_ID": f"{t.rank}",
                "PADDLE_CURRENT_ENDPOINT": f"{t.endpoint}",
                # rank
                "PADDLE_RANK_IN_NODE": f"{idx}",
                "PADDLE_LOCAL_DEVICE_IDS":
                ",".join([str(g) for g in t.devices]),
            }

            if len(t.devices) > 0:
                proc_env["FLAGS_selected_gpus"] = "%s" % ",".join(
                    [str(g) for g in t.devices])
                proc_env["FLAGS_selected_devices"] = "%s" % ",".join(
                    [str(g) for g in t.devices])

            current_env.update(proc_env)

            if os.environ.get(FASTNLP_GLOBAL_SEED) is None and FASTNLP_GLOBAL_SEED in current_env:
                del current_env[FASTNLP_GLOBAL_SEED]

            if idx != 0:
                # 子进程
                if os.environ.get(FASTNLP_LOG_LEVEL, None) is None:
                    current_env[FASTNLP_LOG_LEVEL] = "warning"
                proc = distributed_open_proc(self.output_from_new_proc, command, current_env, t.rank)
            else:
                # 更新当前的环境变量
                os.environ.update(current_env)

    def get_global_env(self):
        """
        设置分布式训练需要的全局变量，包括：
        1、GLOO 相关的设置
        2、`PADDLE_TRAINERS_NUM` ：所有的进程数目
        3、`PADDLE_TRAINER_ENDPOINTS` ：使用的所有地址及其端口
        4、`PADDLE_WORLD_DEVICE_IDS` ：使用的所有设备
        5、FASTNLP_DISTRIBUTED_CHECK：通过 fastNLP 建立子进程的标志，保存分布式训练使用的设备
        """

        global_envs = copy.copy(os.environ.copy())
        self.gloo_rendezvous_dir = tempfile.mkdtemp()
        # launch中涉及的gloo环境
        global_envs["PADDLE_WITH_GLOO"] = str(os.getenv("PADDLE_WITH_GLOO", "0"))
        global_envs["PADDLE_GLOO_RENDEZVOUS"] = "3"
        global_envs["PADDLE_GLOO_FS_PATH"] = self.gloo_rendezvous_dir
        global_envs["PADDLE_DISTRI_BACKEND"] = "nccl"

        # 通过FNLP初始化的标志
        global_envs[FASTNLP_DISTRIBUTED_CHECK] = f"{','.join([str(g) for g in self.devices])}"

        # 统计全局信息
        device_ids = []
        for t in self.sub_trainers:
            device_ids.append([str(acc) for acc in t.devices])
        world_device_ids = [':'.join(ele) for ele in device_ids]
        # 全局环境变量
        global_envs.update({
            # world_size
            "PADDLE_TRAINERS_NUM": f"{len(self.sub_trainers)}",
            "PADDLE_TRAINER_ENDPOINTS": ",".join(self.endpoints),
            "PADDLE_WORLD_DEVICE_IDS": ",".join(world_device_ids),
        })

        return global_envs

    def set_endpoints(self):
        """
        寻找用户设置的端口或是空闲端口用于分布式训练，参考了 PaddlePaddle 中的 `get_cluster_from_args` 函数
        """
        self.node_ip = "127.0.0.1"

        free_ports = None
        if  os.environ.get("FLAGS_START_PORT") is None:
            free_ports = find_free_ports(len(self.devices))
            if free_ports is not None:
                free_ports = list(free_ports)
        else:
            start_port = int(os.getenv("FLAGS_START_PORT", "6070"))

            free_ports = [
                x for x in range(start_port, start_port + len(self.devices))
            ]

        self.endpoints = ["%s:%d" % (self.node_ip, port) for port in free_ports]

    def get_process_info(self):
        """
        获取各个训练进程的设备、rank 和端口信息，参考 PaddlePaddle 的 `get_cluster` 函数。
        """
        sub_trainers = []
        assert len(self.endpoints) >= len(
            self.devices
        ), "current trainer_endpoints size should be greater equal than acclerators size."

        for i in range(len(self.devices)):
            sub_trainer = SubTrainer(f"{self.endpoints[i]}", i)
            if isinstance(self.devices[i], (list, tuple)):
                sub_trainer.devices.extend(self.devices[i])
            else:
                sub_trainer.devices.append(self.devices[i])

            sub_trainers.append(sub_trainer)

        return sub_trainers

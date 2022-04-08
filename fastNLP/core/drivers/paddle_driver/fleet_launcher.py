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
    和fastnlp的Triainer没有关系，仅用于统计节点内不同训练的一些信息
    """
    def __init__(self, endpoint=None, rank=None):
        self.devices = []
        self.endpoint = endpoint
        self.rank = rank


class FleetLauncher:
    """
    复原了 paddle 的 launch_collective 函数，将其集成到一个类里
    仅支持单机多卡的启动
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

        self.set_endpoints()
        self.sub_trainers = self.get_process_info()

    def launch(self) -> int:
        # 设置环境变量
        self.global_envs = self.get_global_env()
        self.open_subprocess()
        reset_seed()

    def open_subprocess(self):

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
        Reference to `get_cluster_from_args`
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
        Reference to `get_cluster`
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

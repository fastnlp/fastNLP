from typing import Optional
from typing import Union, List
import subprocess
from pathlib import Path

from fastNLP.core.drivers.driver import Driver



def choose_driver(model, driver: Union[str, Driver], device: Optional[Union[int, List[int], str]], **kwargs) -> Driver:
    r"""
    根据输入的参数 'gpus' 的格式来决定具体的工作模式;

    :param model: 运行过程中使用的具体的最原始的模型；
    :param driver: 应当为字符串或者 `Driver` 实例，表示运行中具体使用的训练/评测模式；
    :param device: 具体的形式请参见 `fastNLP.core.drivers.torch_driver.utils.initialize_torch_dirver` 的注释；
    :param kwargs: 其余的传给 `Driver` 的参数；
    """

    # 如果用户直接传进来一个 driver 实例，我们就直接返回回去，目前用户需要自己保证传进来的 driver 的正确性；
    if isinstance(driver, Driver):
        return driver

    if driver in {"torch", "torch_ddp", "fairscale"}:
        from fastNLP.core.drivers.torch_driver.initialize_torch_driver import initialize_torch_driver
        return initialize_torch_driver(driver, device, model, **kwargs)
    elif driver in {"jittor"}:
        from fastNLP.core.drivers.jittor_driver.initialize_jittor_driver import initialize_jittor_driver
        return initialize_jittor_driver(driver, device, model, **kwargs)
    elif driver in {"paddle", "fleet"}:
        from fastNLP.core.drivers.paddle_driver.initialize_paddle_driver import initialize_paddle_driver
        return initialize_paddle_driver(driver, device, model, **kwargs)
    else:
        raise ValueError("Parameter `driver` can only be one of these values: ['torch', 'torch_ddp', 'fairscale', "
                         "'jittor', 'paddle', 'fleet'].")



def distributed_open_proc(output_from_new_proc:str, command:List[str], env_copy:dict, rank:int=None):
    """
    使用 command 通过 subprocess.Popen 开启新的进程。

    :param output_from_new_proc: 可选 ["ignore", "all", "only_error"]，以上三个为特殊关键字，分别表示完全忽略拉起进程的打印输出，
        only_error 表示只打印错误输出流；all 表示子进程的所有输出都打印。如果不为以上的关键字，则表示一个文件夹，将在该文件夹下建立
        两个文件，名称分别为 {rank}_std.log, {rank}_err.log 。原有的文件会被直接覆盖。
    :param command: List[str] 启动的命令
    :param env_copy: 需要注入的环境变量。
    :param rank:
    :return:
    """
    if output_from_new_proc == "all":
        proc = subprocess.Popen(command, env=env_copy)
    elif output_from_new_proc == "only_error":
        proc = subprocess.Popen(command, env=env_copy, stdout=subprocess.DEVNULL)
    elif output_from_new_proc == "ignore":
        proc = subprocess.Popen(command, env=env_copy, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        assert rank is not None
        std_f = open(output_from_new_proc + f'/{rank}_std.log', 'w')
        err_f = open(output_from_new_proc + f'/{rank}_err.log', 'w')
        proc = subprocess.Popen(command, env=env_copy, stdout=std_f, stderr=err_f)
    return proc


def load_model(filepath: Union[str, Path], backend: str = "torch", **kwargs):
    r"""
    对应 `load_model`，用来帮助用户加载之前通过 `load_model` 所保存的模型；

    :param filepath: 加载的文件的位置；
    :param backend: 使用哪种 backend 来加载该 filepath， 目前支持 ["torch", "paddle", "jittor"] 。
    """

    if filepath is None:
        raise ValueError("Parameter `path` can not be None.")

    assert backend is not None, "Parameter `backend` can not be None."

    if backend == "torch":
        import torch
        _res = torch.load(filepath)
        return _res
    elif backend == "jittor":
        raise NotImplementedError
    elif backend == "paddle":
        raise NotImplementedError
    else:
        raise ValueError("Parameter `backend` could only be one of these values: ['torch', 'jittor', 'paddle']")



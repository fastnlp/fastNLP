from typing import List
import subprocess

__all__ = []

def distributed_open_proc(output_from_new_proc:str, command:List[str], env_copy:dict, rank:int=None):
    r"""
    使用 command 通过 subprocess.Popen 开启新的进程。

    :param output_from_new_proc: 可选 ``["ignore", "all", "only_error"]``，以上三个为特殊关键字，分别表示：
        * ``"ignore:`` -- 完全忽略拉起进程的打印输出；
        * ``"only_error"`` -- 表示只打印错误输出流；
        * ``"all"`` -- 子进程的所有输出都打印。
        * 如果不为以上的关键字，则表示一个文件夹，将在该文件夹下建立两个文件，名称分别为 {rank}_std.log, {rank}_err.log 。
          原有的文件会被直接覆盖。
    :param command: 启动的命令
    :param env_copy: 需要注入的环境变量。
    :param rank: global_rank；
    :return: 使用 ``subprocess.Popen`` 打开的进程；
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

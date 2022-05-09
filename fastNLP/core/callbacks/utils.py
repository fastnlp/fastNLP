from typing import Optional, Union, Tuple
import os

from fastNLP.core.log.logger import logger
from difflib import SequenceMatcher
from fastNLP.core.utils.utils import _get_fun_msg


def _get_monitor_value(monitor: Union[callable, str], real_monitor: Optional[str], res: dict) ->Tuple[str, float]:
    """
    从 ``res`` 中寻找 ``monitor`` 并返回。如果 ``monitor`` 没找到则尝试用 ``_real_monitor`` ,若 ``_real_monitor`` 为 ``None``
    则尝试使用 ``monitor`` 的值进行匹配。

    :param monitor:
    :param real_monitor:
    :param res:
    :return: 返回两个值（str, value)，其中str就是最终要到的key，value就是这个key对应的value。如果value为None说明当前results中没有
        找到对应的 monitor
    """
    if len(res) == 0 or monitor is None:
        return monitor, None

    if callable(monitor):
        try:
            monitor_value = monitor(res)
        except BaseException as e:
            logger.error(f"Exception happens when calling customized monitor function:{_get_fun_msg(monitor)}.")
            raise e
        return monitor, monitor_value

    if monitor in res:
        return monitor, res[monitor]

    if real_monitor in res:
        return real_monitor, res[real_monitor]

    pairs = []
    for idx, (key, value) in enumerate(res.items()):
        match_size = _match_length(monitor, key)
        pairs.append((key, value, match_size, idx))

    pairs.sort(key=lambda pair: (pair[2], -pair[3]), reverse=True)
    key, value, match_size = pairs[0][:3]

    return key, value


def _match_length(a:str, b:str)->int:
    """
    需要把长度短的放在前面

    :param a:
    :param b:
    :return:
    """
    short = a if len(a) < len(b) else b
    long = a if len(a)>=len(b) else b
    match = SequenceMatcher(None, short, long).find_longest_match(0, len(short), 0, len(long))
    return match.size


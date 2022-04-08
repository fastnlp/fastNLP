from typing import Optional
from fastNLP.core.log.logger import logger
from difflib import SequenceMatcher


def _get_monitor_value(monitor: str, real_monitor: Optional[str], res: dict) ->(str, float):
    """
    从res中寻找 monitor 并返回。如果 monitor 没找到则尝试用 _real_monitor ,若 _real_monitor 为 None 则尝试使用 monitor 的值进行
        匹配。

    :param monitor:
    :param real_monitor:
    :param res:
    :return: 返回两个值（str, value)，其中str就是最终要到的key，value就是这个key对应的value
    """
    if len(res)==0:
        return monitor, 0

    if monitor in res:
        return monitor, res[monitor]

    pairs = []
    for idx, (key, value) in enumerate(res.items()):
        match = SequenceMatcher(None, key, monitor).find_longest_match(0, len(key), 0, len(monitor))
        pairs.append((key, value, match.size, idx))

    pairs.sort(key=lambda pair: (pair[2], -pair[3]), reverse=True)
    key, value, match_size = pairs[0][:3]

    if real_monitor is not None and real_monitor in res and real_monitor != key:
        # 如果 real_monitor 比新找的更长就继续用之前的。
        match = SequenceMatcher(None, real_monitor, monitor).find_longest_match(0, len(real_monitor), 0, len(monitor))
        if match.size > match_size:
            return real_monitor, res[real_monitor]

        logger.warning(f"We can not find `{monitor}` in the evaluation result (with keys as {list(res.keys())}), "
                       f"we use the `{key}` as the monitor.")
    real_monitor = key
    return real_monitor, value



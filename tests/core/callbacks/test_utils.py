import pytest
from tests.helpers.utils import Capturing
from fastNLP.core.callbacks.utils import _get_monitor_value
from fastNLP.core.log.logger import logger


def test_get_monitor_value():
    logger.set_stdout(stdout='raw')

    # 测试完全匹配
    res = {'f1': 0.2, 'acc#rec': 0.3}
    with Capturing() as output:
        monitor, value = _get_monitor_value(monitor='f1', real_monitor=None, res=res)
    assert monitor == 'f1' and value==0.2
    assert 'We can not find' not in output[0]

    # 测试可以匹配，且选择更靠前的
    res = {'acc#f1': 0.2, 'acc#rec': 0.3, 'add#f':0.4}
    with Capturing() as output:
        monitor, value = _get_monitor_value(monitor='f1', real_monitor=None, res=res)
    assert monitor=='acc#f1' and value==0.2
    assert 'We can not find' in output[0]

    # 测试monitor匹配不上，使用real_monitor
    res = {'acc#f1': 0.2, 'acc#rec': 0.3, 'add#f':0.4}
    with Capturing() as output:
        monitor, value = _get_monitor_value(monitor='acc#f', real_monitor='acc#rec', res=res)
    assert monitor=='acc#rec' and value==0.3
    assert 'We can not find' not in output[0]

    # 测试monitor/real_monitor匹配不上, 重新选择
    res = {'acc#f1': 0.2, 'acc#rec': 0.3, 'add#f':0.4}
    with Capturing() as output:
        monitor, value = _get_monitor_value(monitor='acc#f', real_monitor='acc#r', res=res)
    assert monitor=='acc#f1' and value==0.2
    assert 'We can not find' in output[0]

    # 测试partial的位置
    res = {"acc#acc": 0.52, "loss#loss": 2}
    with Capturing() as output:
        monitor, value = _get_monitor_value(monitor='-loss', real_monitor=None, res=res)
    assert monitor=='loss#loss' and value==2
    assert 'We can not find' in output[0]

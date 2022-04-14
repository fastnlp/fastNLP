from functools import partial

import pytest

from fastNLP.core.utils.utils import auto_param_call, _check_valid_parameters_number, _get_fun_msg
from fastNLP.core.metrics import Metric



class TestAutoParamCall:
    def test_basic(self):
        def fn(x):
            return x
        x = {'x': 3, 'y': 4}
        r = auto_param_call(fn, x)
        assert r==3

        xs = []
        for i in range(10):
            xs.append({f'x{i}': i})
        def fn(x0, x1, x2, x3):
            return x0 + x1 + x2 + x3
        r = auto_param_call(fn, *xs)
        assert r == 0 + 1+ 2+ 3

        def fn(chongfu1, chongfu2, buChongFu):
            pass
        with pytest.raises(BaseException) as exc_info:
            auto_param_call(fn, {'chongfu1': 3, "chongfu2":4, 'buChongFu':2},
                            {'chongfu1': 1, 'chongfu2':2, 'buChongFu':2})
        assert 'The following key present in several inputs' in exc_info.value.args[0]
        assert 'chongfu1' in exc_info.value.args[0] and  'chongfu2' in exc_info.value.args[0]

        # 没用到不报错
        def fn(chongfu1, buChongFu):
            pass
        auto_param_call(fn, {'chongfu1': 1, "chongfu2":4, 'buChongFu':2},
                        {'chongfu1': 1, 'chongfu2':2, 'buChongFu':2})

        # 可以定制signature_fn
        def fn1(**kwargs):
            kwargs.pop('x')
            kwargs.pop('y')
            assert len(kwargs)==0
        def fn(x, y):
            pass
        x = {'x': 3, 'y': 4}
        r = auto_param_call(fn1, x, signature_fn=fn)

        # 没提供的时候报错
        def fn(meiti1, meiti2, tigong):
            pass
        with pytest.raises(BaseException) as exc_info:
            auto_param_call(fn, {'tigong':1})
        assert 'meiti1' in exc_info.value.args[0] and 'meiti2' in exc_info.value.args[0]

        # 默认值替换
        def fn(x, y=100):
            return x + y
        r = auto_param_call(fn, {'x': 10, 'y': 20})
        assert r==30
        assert auto_param_call(fn, {'x': 10, 'z': 20})==110

        # 测试mapping的使用
        def fn(x, y=100):
            return x + y
        r = auto_param_call(fn, {'x1': 10, 'y1': 20}, mapping={'x1': 'x', 'y1': 'y', 'meiyong': 'meiyong'})
        assert r==30

        # 测试不需要任何参数
        def fn():
            return 1
        assert 1 == auto_param_call(fn, {'x':1})

        # 测试调用类的方法没问题
        assert 2==auto_param_call(self.call_this, {'x':1 ,'y':1})
        assert 2==auto_param_call(self.call_this, {'x':1,'y':1, 'z':1},mapping={'z': 'self'})

    def test_msg(self):
        with pytest.raises(BaseException) as exc_info:
            auto_param_call(self.call_this, {'x':1})
        assert 'TestAutoParamCall.call_this' in exc_info.value.args[0]

        with pytest.raises(BaseException) as exc_info:
            auto_param_call(call_this_for_auto_param_call, {'x':1})
        assert __file__ in exc_info.value.args[0]
        assert 'call_this_for_auto_param_call' in exc_info.value.args[0]

        with pytest.raises(BaseException) as exc_info:
            auto_param_call(self.call_this_two, {'x':1})
        assert __file__ in exc_info.value.args[0]

        with pytest.raises(BaseException) as exc_info:
            auto_param_call(call_this_for_auto_param_call, {'x':1}, signature_fn=self.call_this)
        assert 'TestAutoParamCall.call_this' in exc_info.value.args[0]  # 应该是signature的信息

    def call_this(self, x, y):
        return x + y

    def call_this_two(self, x, y, z=pytest, **kwargs):
        return x + y

    def test_metric_auto_param_call(self):
        metric = AutoParamCallMetric()
        with pytest.raises(BaseException):
            auto_param_call(metric.update, {'y':1}, signature_fn=metric.update.__wrapped__)


class AutoParamCallMetric(Metric):
    def update(self, x):
        pass


def call_this_for_auto_param_call(x, y):
    return x + y


class TestCheckNumberOfParameters:
    def test_validate_every(self):
        def validate_every(trainer):
            pass
        _check_valid_parameters_number(validate_every, expected_params=['trainer'])

        # 无默认值，多了报错
        def validate_every(trainer, other):
            pass
        with pytest.raises(RuntimeError) as exc_info:
            _check_valid_parameters_number(validate_every, expected_params=['trainer'])
        assert "2 parameters" in exc_info.value.args[0]
        print(exc_info.value.args[0])

        # 有默认值ok
        def validate_every(trainer, other=1):
            pass
        _check_valid_parameters_number(validate_every, expected_params=['trainer'])

        # 参数多了
        def validate_every(trainer):
            pass
        with pytest.raises(RuntimeError) as exc_info:
            _check_valid_parameters_number(validate_every, expected_params=['trainer', 'other'])
        assert "accepts 1 parameters" in exc_info.value.args[0]
        print(exc_info.value.args[0])

        # 使用partial
        def validate_every(trainer, other):
            pass
        _check_valid_parameters_number(partial(validate_every, other=1), expected_params=['trainer'])
        _check_valid_parameters_number(partial(validate_every, other=1), expected_params=['trainer', 'other'])
        with pytest.raises(RuntimeError) as exc_info:
            _check_valid_parameters_number(partial(validate_every, other=1), expected_params=['trainer', 'other', 'more'])
        assert 'accepts 2 parameters' in exc_info.value.args[0]
        print(exc_info.value.args[0])

        # 如果存在 *args 或 *kwargs 不报错多的
        def validate_every(trainer, *args):
            pass
        _check_valid_parameters_number(validate_every, expected_params=['trainer', 'other', 'more'])

        def validate_every(trainer, **kwargs):
            pass
        _check_valid_parameters_number(partial(validate_every, trainer=1), expected_params=['trainer', 'other', 'more'])

        # class 的方法删掉self
        class InnerClass:
            def demo(self, x):
                pass

            def no_param(self):
                pass

            def param_kwargs(self, **kwargs):
                pass

        inner = InnerClass()
        with pytest.raises(RuntimeError) as exc_info:
            _check_valid_parameters_number(inner.demo, expected_params=['trainer', 'other', 'more'])
        assert 'accepts 1 parameters' in exc_info.value.args[0]

        _check_valid_parameters_number(inner.demo, expected_params=['trainer'])


def test_get_fun_msg():
    def demo(x):
        pass

    print(_get_fun_msg(_get_fun_msg))
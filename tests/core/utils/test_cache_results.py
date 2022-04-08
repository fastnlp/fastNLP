import time
import os
import pytest
from subprocess import Popen, PIPE
from io import StringIO
import sys

from fastNLP.core.utils.cache_results import cache_results
from tests.helpers.common.utils import check_time_elapse

from fastNLP.core import synchronize_safe_rm


def get_subprocess_results(cmd):
    pipe = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    output, err = pipe.communicate()
    if output:
        output = output.decode('utf8')
    else:
        output = ''
    if err:
        err = err.decode('utf8')
    else:
        err = ''
    res = output + err
    return res


class Capturing(list):
    # 用来捕获当前环境中的stdout和stderr，会将其中stderr的输出拼接在stdout的输出后面
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = self._stringio = StringIO()
        sys.stderr = self._stringioerr = StringIO()
        return self

    def __exit__(self, *args):
        self.append(self._stringio.getvalue() + self._stringioerr.getvalue())
        del self._stringio, self._stringioerr    # free up some memory
        sys.stdout = self._stdout
        sys.stderr = self._stderr


class TestCacheResults:
    def test_cache_save(self):
        cache_fp = 'demo.pkl'
        try:
            @cache_results(cache_fp)
            def demo():
                time.sleep(1)
                return 1

            res = demo()
            with check_time_elapse(1, op='lt'):
                res = demo()

        finally:
            synchronize_safe_rm(cache_fp)

    def test_cache_save_refresh(self):
        cache_fp = 'demo.pkl'
        try:
            @cache_results(cache_fp, _refresh=True)
            def demo():
                time.sleep(1.5)
                return 1

            res = demo()
            with check_time_elapse(1, op='ge'):
                res = demo()
        finally:
            synchronize_safe_rm(cache_fp)

    def test_cache_no_func_change(self):
        cache_fp = os.path.abspath('demo.pkl')
        try:
            @cache_results(cache_fp)
            def demo():
                time.sleep(2)
                return 1

            with check_time_elapse(1, op='gt'):
                res = demo()

            @cache_results(cache_fp)
            def demo():
                time.sleep(2)
                return 1

            with check_time_elapse(1, op='lt'):
                res = demo()
        finally:
            synchronize_safe_rm('demo.pkl')

    def test_cache_func_change(self, capsys):
        cache_fp = 'demo.pkl'
        try:
            @cache_results(cache_fp)
            def demo():
                time.sleep(2)
                return 1

            with check_time_elapse(1, op='gt'):
                res = demo()

            @cache_results(cache_fp)
            def demo():
                time.sleep(1)
                return 1

            with check_time_elapse(1, op='lt'):
                with Capturing() as output:
                    res = demo()
                assert 'is different from its last cache' in output[0]

            # 关闭check_hash应该不warning的
            with check_time_elapse(1, op='lt'):
                with Capturing() as output:
                    res = demo(_check_hash=0)
                assert 'is different from its last cache' not in output[0]

        finally:
            synchronize_safe_rm('demo.pkl')

    def test_cache_check_hash(self):
        cache_fp = 'demo.pkl'
        try:
            @cache_results(cache_fp, _check_hash=False)
            def demo():
                time.sleep(2)
                return 1

            with check_time_elapse(1, op='gt'):
                res = demo()

            @cache_results(cache_fp, _check_hash=False)
            def demo():
                time.sleep(1)
                return 1

            # 默认不会check
            with check_time_elapse(1, op='lt'):
                with Capturing() as output:
                    res = demo()
                assert 'is different from its last cache' not in output[0]

            # check也可以
            with check_time_elapse(1, op='lt'):
                with Capturing() as output:
                    res = demo(_check_hash=True)
                assert 'is different from its last cache' in output[0]

        finally:
            synchronize_safe_rm('demo.pkl')

    # 外部 function 改变也会 导致改变
    def test_refer_fun_change(self):
        cache_fp = 'demo.pkl'
        test_type = 'func_refer_fun_change'
        try:
            with check_time_elapse(3, op='gt'):
                cmd = f'python {__file__} --cache_fp {cache_fp} --test_type {test_type} --turn 0'
                res = get_subprocess_results(cmd)

            # 引用的function没有变化
            with check_time_elapse(2, op='lt'):
                cmd = f'python {__file__} --cache_fp {cache_fp} --test_type {test_type} --turn 0'
                res = get_subprocess_results(cmd)
                assert 'Read cache from' in res
                assert 'is different from its last cache' not in res

            # 引用的function有变化
            with check_time_elapse(2, op='lt'):
                cmd = f'python {__file__} --cache_fp {cache_fp} --test_type {test_type} --turn 1'
                res = get_subprocess_results(cmd)
                assert 'is different from its last cache' in res

        finally:
            synchronize_safe_rm(cache_fp)

    # 外部 method 改变也会 导致改变
    def test_refer_class_method_change(self):
        cache_fp = 'demo.pkl'
        test_type = 'refer_class_method_change'
        try:
            with check_time_elapse(3, op='gt'):
                cmd = f'python {__file__} --cache_fp {cache_fp} --test_type {test_type} --turn 0'
                res = get_subprocess_results(cmd)

            # 引用的class没有变化
            with check_time_elapse(2, op='lt'):
                cmd = f'python {__file__} --cache_fp {cache_fp} --test_type {test_type} --turn 0'
                res = get_subprocess_results(cmd)
                assert 'Read cache from' in res
                assert 'is different from its last cache' not in res

            # 引用的class有变化
            with check_time_elapse(2, op='lt'):
                cmd = f'python {__file__} --cache_fp {cache_fp} --test_type {test_type} --turn 1'
                res = get_subprocess_results(cmd)
                assert 'is different from its last cache' in res

        finally:
            synchronize_safe_rm(cache_fp)

    def test_duplicate_keyword(self):
        with pytest.raises(RuntimeError):
            @cache_results(None)
            def func_verbose(a, _verbose):
                pass

            func_verbose(0, 1)
        with pytest.raises(RuntimeError):
            @cache_results(None)
            def func_cache(a, _cache_fp):
                pass

            func_cache(1, 2)
        with pytest.raises(RuntimeError):
            @cache_results(None)
            def func_refresh(a, _refresh):
                pass

            func_refresh(1, 2)

        with pytest.raises(RuntimeError):
            @cache_results(None)
            def func_refresh(a, _check_hash):
                pass

            func_refresh(1, 2)

    def test_create_cache_dir(self):
        @cache_results('demo/demo.pkl')
        def cache():
            return 1, 2

        try:
            results = cache()
            assert (1, 2) == results
        finally:
            synchronize_safe_rm('demo/')

    def test_result_none_error(self):
        @cache_results('demo.pkl')
        def cache():
            pass

        try:
            with pytest.raises(RuntimeError):
                results = cache()
        finally:
            synchronize_safe_rm('demo.pkl')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_type', type=str, default='refer_class_method_change')
    parser.add_argument('--turn', type=int, default=1)
    parser.add_argument('--cache_fp', type=str, default='demo.pkl')
    args = parser.parse_args()

    test_type = args.test_type
    cache_fp = args.cache_fp
    turn = args.turn

    if test_type == 'func_refer_fun_change':
        if turn == 0:
            def demo():
                b = 1
                return b
        else:
            def demo():
                b = 2
                return b

        @cache_results(cache_fp)
        def demo_refer_other_func():
            time.sleep(3)
            b = demo()
            return b

        res = demo_refer_other_func()

    if test_type == 'refer_class_method_change':
        print(f"Turn:{turn}")
        if turn == 0:
            from helper_for_cache_results_1 import Demo
        else:
            from helper_for_cache_results_2 import Demo

        demo = Demo()
        # import pdb
        # pdb.set_trace()
        @cache_results(cache_fp)
        def demo_func():
            time.sleep(3)
            b = demo.demo()
            return b

        res = demo_func()


import os
import pytest
import subprocess
from io import StringIO
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))

from fastNLP.core.utils.cache_results import cache_results
from fastNLP.core import rank_zero_rm


def get_subprocess_results(cmd):
    output = subprocess.check_output(cmd, shell=True)
    return output.decode('utf8')


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
                print("¥")
                return 1
            res = demo()
            with Capturing() as output:
                res = demo()
            assert '¥' not in output[0]

        finally:
            rank_zero_rm(cache_fp)

    def test_cache_save_refresh(self):
        cache_fp = 'demo.pkl'
        try:
            @cache_results(cache_fp, _refresh=True)
            def demo():
                print("¥")
                return 1

            res = demo()
            with Capturing() as output:
                res = demo()
            assert '¥' in output[0]
        finally:
            rank_zero_rm(cache_fp)

    def test_cache_no_func_change(self):
        cache_fp = os.path.abspath('demo.pkl')
        try:
            @cache_results(cache_fp)
            def demo():
                print('¥')
                return 1

            with Capturing() as output:
                res = demo()
            assert '¥' in output[0]

            @cache_results(cache_fp)
            def demo():
                print('¥')
                return 1

            with Capturing() as output:
                res = demo()
            assert '¥' not in output[0]
        finally:
            rank_zero_rm('demo.pkl')

    def test_cache_func_change(self, capsys):
        cache_fp = 'demo.pkl'
        try:
            @cache_results(cache_fp)
            def demo():
                print('¥')
                return 1

            with Capturing() as output:
                res = demo()
            assert '¥' in output[0]

            @cache_results(cache_fp)
            def demo():
                print('¥¥')
                return 1

            with Capturing() as output:
                res = demo()
            assert 'different' in output[0]
            assert '¥' not in output[0]

            # 关闭check_hash应该不warning的
            with Capturing() as output:
                res = demo(_check_hash=0)
            assert 'different' not in output[0]
            assert '¥' not in output[0]

        finally:
            rank_zero_rm('demo.pkl')

    def test_cache_check_hash(self):
        cache_fp = 'demo.pkl'
        try:
            @cache_results(cache_fp, _check_hash=False)
            def demo():
                print('¥')
                return 1

            with Capturing() as output:
                res = demo(_check_hash=0)
            assert '¥' in output[0]

            @cache_results(cache_fp, _check_hash=False)
            def demo():
                print('¥¥')
                return 1

            # 默认不会check
            with Capturing() as output:
                res = demo()
            assert 'different' not in output[0]
            assert '¥' not in output[0]

            # check也可以
            with Capturing() as output:
                res = demo(_check_hash=True)
            assert 'different' in output[0]
            assert '¥' not in output[0]

        finally:
            rank_zero_rm('demo.pkl')

    # 外部 function 改变也会 导致改变
    def test_refer_fun_change(self):
        cache_fp = 'demo.pkl'
        test_type = 'func_refer_fun_change'
        try:
            cmd = f'python {__file__} --cache_fp {cache_fp} --test_type {test_type} --turn 0'
            res = get_subprocess_results(cmd)
            assert "¥" in res
            # 引用的function没有变化
            cmd = f'python {__file__} --cache_fp {cache_fp} --test_type {test_type} --turn 0'
            res = get_subprocess_results(cmd)
            assert "¥" not in res

            assert 'Read' in res
            assert 'different' not in res

            # 引用的function有变化
            cmd = f'python {__file__} --cache_fp {cache_fp} --test_type {test_type} --turn 1'
            res = get_subprocess_results(cmd)
            assert "¥" not in res
            assert 'different' in res

        finally:
            rank_zero_rm(cache_fp)

    # 外部 method 改变也会 导致改变
    def test_refer_class_method_change(self):
        cache_fp = 'demo.pkl'
        test_type = 'refer_class_method_change'
        try:
            cmd = f'python {__file__} --cache_fp {cache_fp} --test_type {test_type} --turn 0'
            res = get_subprocess_results(cmd)
            assert "¥" in res

            # 引用的class没有变化
            cmd = f'python {__file__} --cache_fp {cache_fp} --test_type {test_type} --turn 0'
            res = get_subprocess_results(cmd)
            assert 'Read' in res
            assert 'different' not in res
            assert "¥" not in res

            cmd = f'python {__file__} --cache_fp {cache_fp} --test_type {test_type} --turn 1'
            res = get_subprocess_results(cmd)
            assert 'different' in res
            assert "¥" not in res

        finally:
            rank_zero_rm(cache_fp)

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
            rank_zero_rm('demo/')

    def test_result_none_error(self):
        @cache_results('demo.pkl')
        def cache():
            pass

        try:
            with pytest.raises(RuntimeError):
                results = cache()
        finally:
            rank_zero_rm('demo.pkl')


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
            b = demo()
            print("¥")
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
            print("¥")
            b = demo.demo()
            return b

        res = demo_func()


from typing import Callable
__all__ = [
    "indice_collate_wrapper"
]


def indice_collate_wrapper(func:Callable):
    """
    其功能是封装一层collate_fn,将dataset取到的tuple数据分离开，将idx打包为indices。

    :param func: 需要修饰的函数
    :return:
    """
    if hasattr(func, '__name__') and func.__name__ == '_indice_collate_wrapper':  # 如果已经被包裹过了
       return func

    def _indice_collate_wrapper(tuple_data):  # 这里不能使用 functools.wraps ，否则会检测不到
        indice, ins_list = [], []
        for idx, ins in tuple_data:
            indice.append(idx)
            ins_list.append(ins)
        return indice, func(ins_list)
    _indice_collate_wrapper.__wrapped__ = func  # 记录对应的

    return _indice_collate_wrapper


if __name__ == '__main__':
    def demo(*args, **kwargs):
        pass

    d = indice_collate_wrapper(demo)

    print(d.__name__)
    print(d.__wrapped__)
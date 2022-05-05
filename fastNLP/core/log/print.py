__all__ = [
    'print'
]

from .logger import logger


def print(*args, sep=' ', end='\n', file=None, flush=False):
    """
    用来重定向 print 函数至 logger.info 的函数。

    Example:
        from fastNLP import print

        print("This is a test")  # 等价于调用了 logger.info("This is a test")

    :param args: 需要打印的内容
    :param sep: 存在多个输入时，使用的间隔。
    :param end: 该参数在当前设置无意义，因为结尾一定会被加入 \n 。
    :param file: 该参数无意义。
    :param flush: 该参数无意义。
    :return:
    """
    line = sep.join(map(str, args))
    logger.info(line)
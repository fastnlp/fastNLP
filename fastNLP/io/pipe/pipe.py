"""undocumented"""

__all__ = [
    "Pipe",
]

from .. import DataBundle


class Pipe:
    """
    .. todo::
        doc

    """
    def process(self, data_bundle: DataBundle) -> DataBundle:
        """
        对输入的DataBundle进行处理，然后返回该DataBundle。

        :param ~fastNLP.DataBundle data_bundle: 需要处理的DataBundle对象
        :return:
        """
        raise NotImplementedError

    def process_from_file(self, paths) -> DataBundle:
        """
        传入文件路径，生成处理好的DataBundle对象。paths支持的路径形式可以参考 ：:meth:`fastNLP.io.Loader.load()`

        :param paths:
        :return: DataBundle
        """
        raise NotImplementedError

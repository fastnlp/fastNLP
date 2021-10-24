r"""undocumented"""

__all__ = [
    "Pipe",
]

from .. import DataBundle


class Pipe:
    r"""
    Pipe是fastNLP中用于处理DataBundle的类，但实际是处理DataBundle中的DataSet。所有Pipe都会在其process()函数的文档中指出该Pipe可处理的DataSet应该具备怎样的格式；在Pipe
    文档中说明该Pipe返回后DataSet的格式以及其field的信息；以及新增的Vocabulary的信息。

    一般情况下Pipe处理包含以下的几个过程，(1)将raw_words或raw_chars进行tokenize以切分成不同的词或字;
    (2) 再建立词或字的 :class:`~fastNLP.Vocabulary` , 并将词或字转换为index; (3)将target列建立词表并将target列转为index;

    Pipe中提供了两个方法

    -process()函数，输入为DataBundle
    -process_from_file()函数，输入为对应Loader的load函数可接受的类型。

    """
    
    def process(self, data_bundle: DataBundle) -> DataBundle:
        r"""
        对输入的DataBundle进行处理，然后返回该DataBundle。

        :param ~fastNLP.DataBundle data_bundle: 需要处理的DataBundle对象
        :return:
        """
        raise NotImplementedError

    def process_from_file(self, paths) -> DataBundle:
        r"""
        传入文件路径，生成处理好的DataBundle对象。paths支持的路径形式可以参考 ：:meth:`fastNLP.io.Loader.load()`

        :param paths:
        :return: DataBundle
        """
        raise NotImplementedError

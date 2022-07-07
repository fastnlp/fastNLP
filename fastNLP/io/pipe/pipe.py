__all__ = [
    "Pipe",
]

from fastNLP.io.data_bundle import DataBundle


class Pipe:
    r"""
    :class:`Pipe` 是 **fastNLP** 中用于处理 :class:`~fastNLP.io.DataBundle` 的类，但实际是处理其中的 :class:`~fastNLP.core.DataSet` 。
    所有 ``Pipe`` 都会在其 :meth:`process` 函数的文档中指出该 ``Pipe`` 可处理的 :class:`~fastNLP.core.DataSet` 应该具备怎样的格式；在
    ``Pipe`` 文档中说明该 ``Pipe`` 返回后 :class:`~fastNLP.core.DataSet` 的格式以及其 field 的信息；以及新增的 :class:`~fastNLP.core.Vocabulary` 
    的信息。

    一般情况下 **Pipe** 处理包含以下的几个过程：
    
        1. 将 ``raw_words`` 或 ``raw_chars`` 进行 tokenize 以切分成不同的词或字；
        2. 建立词或字的 :class:`~fastNLP.core.Vocabulary` ，并将词或字转换为 index；
        3. 将 ``target`` 列建立词表并将 ``target`` 列转为 index；

    **Pipe** 中提供了两个方法：

        - :meth:`process` 函数，输入为 :class:`~fastNLP.io.DataBundle`
        - :meth:`process_from_file` 函数，输入为对应 :meth:`fastNLP.io.Loader.load` 函数可接受的类型。

    """
    
    def process(self, data_bundle: DataBundle) -> DataBundle:
        r"""
        对输入的 ``data_bundle`` 进行处理，然后返回该 ``data_bundle``

        :param data_bundle:
        :return: 处理后的 ``data_bundle``
        """
        raise NotImplementedError

    def process_from_file(self, paths: str) -> DataBundle:
        r"""
        传入文件路径，生成处理好的 :class:`~fastNLP.io.DataBundle` 对象。``paths`` 支持的路径形式可以参考 :meth:`fastNLP.io.Loader.load`

        :param paths:
        :return:
        """
        raise NotImplementedError

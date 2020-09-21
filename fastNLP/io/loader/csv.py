r"""undocumented"""

__all__ = [
    "CSVLoader",
]

from .loader import Loader
from ..file_reader import _read_csv
from ...core.dataset import DataSet
from ...core.instance import Instance


class CSVLoader(Loader):
    r"""
    读取CSV格式的数据集, 返回 ``DataSet`` 。

    """

    def __init__(self, headers=None, sep=",", dropna=False):
        r"""
        
        :param List[str] headers: CSV文件的文件头.定义每一列的属性名称,即返回的DataSet中`field`的名称
            若为 ``None`` ,则将读入文件的第一行视作 ``headers`` . Default: ``None``
        :param str sep: CSV文件中列与列之间的分隔符. Default: ","
        :param bool dropna: 是否忽略非法数据,若 ``True`` 则忽略,若 ``False`` ,在遇到非法数据时,抛出 ``ValueError`` .
            Default: ``False``
        """
        super().__init__()
        self.headers = headers
        self.sep = sep
        self.dropna = dropna

    def _load(self, path):
        ds = DataSet()
        for idx, data in _read_csv(path, headers=self.headers,
                                   sep=self.sep, dropna=self.dropna):
            ds.append(Instance(**data))
        return ds


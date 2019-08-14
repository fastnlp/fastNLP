
from .loader import Loader
from ...core import DataSet, Instance


class CWSLoader(Loader):
    """
    分词任务数据加载器，
    SigHan2005的数据可以用xxx下载并预处理

    CWSLoader支持的数据格式为，一行一句话，不同词之间用空格隔开, 例如：

    Example::

        上海 浦东 开发 与 法制 建设 同步
        新华社 上海 二月 十日 电 （ 记者 谢金虎 、 张持坚 ）
        ...

    该Loader读取后的DataSet具有如下的结构

    .. csv-table::
       :header: "raw_words"

       "上海 浦东 开发 与 法制 建设 同步"
       "新华社 上海 二月 十日 电 （ 记者 谢金虎 、 张持坚 ）"
       "..."
    """
    def __init__(self):
        super().__init__()

    def _load(self, path:str):
        ds = DataSet()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    ds.append(Instance(raw_words=line))
        return ds

    def download(self, output_dir=None):
        raise RuntimeError("You can refer {} for sighan2005's data downloading.")

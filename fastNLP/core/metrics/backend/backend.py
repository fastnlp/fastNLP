from ..utils import AggregateMethodError


class Backend:
    """
    Backend 及其子类的所有方法都必须是无状态的。

    """

    def __init__(self):
        self._specified = False

    def aggregate(self, tensor, method: str):
        """
        聚集结果，并根据 method 计算后，返回结果
        :param tensor: 传入的张量
        :param method: 聚合的方法
        """
        if method is not None:
            return AggregateMethodError(should_have_aggregate_method=False, only_warn=True)

        return tensor

    def create_tensor(self, value: float):
        """
        创建tensor，并且填入value作为值

        :param value: 需要初始化的 value 值
        """
        return value

    def fill_value(self, tensor, value: float):
        """
        将tensor的值设置为value

        :param tensor: 传进来的张量
        :param value: 需要填充的值
        """
        return value

    def get_scalar(self, tensor) -> float:
        """
        tensor的saclar值

        :param tensor: 传入的张量
        :return:
        """
        return tensor

    def is_specified(self) -> bool:
        """
        判断是否是某种框架的 backend

        :return:
        """
        return self._specified

    def tensor2numpy(self, tensor):
        """
        将 tensor 转为 numpy

        :param tensor: 传入的张量
        :return:
        """
        return tensor

    def move_tensor_to_device(self, tensor, device):
        """
        将张量移动到某个设备上

        :param tensor: 传入的张量
        :param device: 设备号， 一般为 ``'cpu'``, ``'cuda:0'`` 等。
        """
        return tensor

    def all_gather_object(self, obj, group=None):
        """
        给定 obj 将各个 rank 上的 obj 汇总到每个 obj 上。返回一个 list 对象，里面依次为各个 rank 对应的 obj 。

        :param obj:
        :param group:
        :return:
        """
        if self.__class__.__name__ == 'AutoBackend':
            raise RuntimeError("fastNLP cannot determine the backend automatically, please pass in the backend through "
                               "initialization.")

        raise NotImplementedError(f"all_gather_object() function is not implemented for {self.__class__.__name__}.")


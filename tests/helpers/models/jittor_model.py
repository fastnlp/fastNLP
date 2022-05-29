from fastNLP.envs.imports import _NEED_IMPORT_JITTOR
if _NEED_IMPORT_JITTOR:
    from jittor import Module, nn
else:
    from fastNLP.core.utils.dummy_class import DummyClass as Module

class JittorNormalModel_Classification_1(Module):
    """
    基础的 jittor 分类模型
    """
    def __init__(self, num_labels, feature_dimension):
        super(JittorNormalModel_Classification_1, self).__init__()
        self.num_labels = num_labels

        self.linear1 = nn.Linear(in_features=feature_dimension, out_features=64)
        self.ac1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=64, out_features=32)
        self.ac2 = nn.ReLU()
        self.output = nn.Linear(in_features=32, out_features=num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def execute(self, x):
        x = self.ac1(self.linear1(x))
        x = self.ac2(self.linear2(x))
        x = self.output(x)
        return x

    def train_step(self, x, y):
        x = self(x)
        return {"loss": self.loss_fn(x, y)}

    def evaluate_step(self, x, y):

        x = self(x)
        return {"pred": x, "target": y.reshape((-1,))}


class JittorNormalModel_Classification_2(Module):
    """
    基础的 jittor 分类模型，只实现 execute 函数测试用户自己初始化了分布式的场景
    """
    def __init__(self, num_labels, feature_dimension):
        super(JittorNormalModel_Classification_2, self).__init__()
        self.num_labels = num_labels

        self.linear1 = nn.Linear(in_features=feature_dimension, out_features=64)
        self.ac1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=64, out_features=32)
        self.ac2 = nn.ReLU()
        self.output = nn.Linear(in_features=32, out_features=num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def execute(self, x, y):
        x = self.ac1(self.linear1(x))
        x = self.ac2(self.linear2(x))
        x = self.output(x)
        return {"loss": self.loss_fn(x, y), "pred": x, "target": y.reshape((-1,))}

import paddle
import paddle.nn as nn

class PaddleNormalModel_Classification(paddle.nn.Layer):
    """
    基础的paddle分类模型
    """
    def __init__(self, num_labels, feature_dimension):
        super(PaddleNormalModel_Classification, self).__init__()
        self.num_labels = num_labels

        self.linear1 = nn.Linear(in_features=feature_dimension, out_features=64)
        self.ac1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features=64, out_features=32)
        self.ac2 = nn.ReLU()
        self.output = nn.Linear(in_features=32, out_features=num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.ac1(self.linear1(x))
        x = self.ac2(self.linear2(x))
        x = self.output(x)
        return x

    def train_step(self, x, y):
        x = self(x)
        return {"loss": self.loss_fn(x, y)}

    def validate_step(self, x, y):

        x = self(x)
        return {"pred": x, "target": y.reshape((-1,))}

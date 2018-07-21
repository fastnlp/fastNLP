import aggregation
import decoder
import encoder


class Input(object):
    def __init__(self):
        pass


class Trainer(object):
    def __init__(self, input, target, truth):
        pass

    def train(self):
        pass


def test_keras_like():
    data_train, label_train = dataLoader("./data_path")

    x = Input()
    x = encoder.LSTM(input=x)
    x = aggregation.max_pool(input=x)
    y = decoder.CRF(input=x)

    trainer = Trainer(input=data_train, target=y, truth=label_train)
    trainer.train()

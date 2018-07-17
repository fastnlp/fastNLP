def test_trainer():
    Config = namedtuple("config", ["epochs", "validate", "save_when_better"])
    train_config = Config(epochs=5, validate=True, save_when_better=True)
    trainer = Trainer(train_config)

    net = ToyModel()
    data = np.random.rand(20, 6)
    dev_data = np.random.rand(20, 6)
    trainer.train(net, data, dev_data)


if __name__ == "__main__":
    test_trainer()


import torch


class API:
    def __init__(self):
        self.pipeline = None
        self.model = None

    def predict(self):
        pass

    def load(self, name):
        _dict = torch.load(name)
        self.pipeline = _dict['pipeline']
        self.model = _dict['model']

    def save(self, path):
        _dict = {'pipeline': self.pipeline,
                 'model': self.model}
        torch.save(_dict, path)

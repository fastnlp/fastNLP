
import _pickle


class API:
    def __init__(self):
        self.pipeline = None
        self.model = None

    def predict(self):
        pass

    def load(self, name):
        _dict = _pickle.load(name)
        self.pipeline = _dict['pipeline']
        self.model = _dict['model']

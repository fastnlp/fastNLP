

class BasePreprocess(object):


    def __init__(self, data, pickle_path):
        super(BasePreprocess, self).__init__()
        self.data = data
        self.pickle_path = pickle_path
        if not self.pickle_path.endswith('/'):
            self.pickle_path = self.pickle_path + '/'

    def word2id(self):
        raise NotImplementedError

    def id2word(self):
        raise NotImplementedError

    def class2id(self):
        raise NotImplementedError

    def id2class(self):
        raise NotImplementedError

    def embedding(self):
        raise NotImplementedError

    def data_train(self):
        raise NotImplementedError

    def data_dev(self):
        raise NotImplementedError

    def data_test(self):
        raise NotImplementedError

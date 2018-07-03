

class BasePreprocess(object):


    def __init__(self, data, pickle_path):
        super(BasePreprocess, self).__init__()
        self.data = data
        self.pickle_path = pickle_path
        if not self.pickle_path.endswith('/'):
            self.pickle_path = self.pickle_path + '/'

    def word2id(self):
        pass

    def id2word(self):
        pass

    def class2id(self):
        pass

    def id2class(self):
        pass

    def embedding(self):
        pass

    def data_train(self):
        pass

    def data_dev(self):
        pass

    def data_test(self):
        pass
import torch

class Field(object):
    def __init__(self, is_target: bool):
        self.is_target = is_target

    def index(self, vocab):
        pass
    
    def get_length(self):
        pass

    def to_tensor(self, padding_length):
        pass
        

class TextField(Field):
    def __init__(self, text: list, is_target):
        """
        :param list text:
        """
        super(TextField, self).__init__(is_target)
        self.text = text
        self._index = None

    def index(self, vocab):
        if self._index is None:
            self._index = [vocab[c] for c in self.text]
        else:
            print('error')
        return self._index

    def get_length(self):
        return len(self.text)

    def to_tensor(self, padding_length: int):
        pads = []
        if self._index is None:
            print('error')
        if padding_length > self.get_length():
            pads = [0 for i in range(padding_length - self.get_length())]
        # (length, )
        return torch.LongTensor(self._index + pads)
        

class LabelField(Field):
    def __init__(self, label, is_target=True):
        super(LabelField, self).__init__(is_target)
        self.label = label
        self._index = None
    
    def get_length(self):
        return 1

    def index(self, vocab):
        if self._index is None:
            self._index = vocab[self.label]
        else:
            pass
        return self._index
    
    def to_tensor(self, padding_length):
        if self._index is None:
            return torch.LongTensor([self.label])
        else:
            return torch.LongTensor([self._index])

if __name__ == "__main__":
    tf = TextField("test the code".split())
    

from collections import defaultdict


class DataSet(list):
    def __init__(self, name="", instances=None):
        list.__init__([])
        self.name = name
        if instances is not None:
            self.extend(instances)
    
    def index_all(self, vocab):
        for ins in self:
            ins.index_all(vocab)
    
    def index_field(self, field_name, vocab):
        for ins in self:
            ins.index_field(field_name, vocab)

    def to_tensor(self, idx: int, padding_length: dict):
        ins = self[idx]
        return ins.to_tensor(padding_length)
    
    def get_length(self):
        lengths = defaultdict(list)
        for ins in self:
            for field_name, field_length in ins.get_length().items():
                lengths[field_name].append(field_length)
        return lengths


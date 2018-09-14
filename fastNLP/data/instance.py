class Instance(object):
    def __init__(self, **fields):
        self.fields = fields
        self.has_index = False
        self.indexes = {}

    def add_field(self, field_name, field):
        self.fields[field_name] = field
    
    def get_length(self):
        length = {name : field.get_length() for name, field in self.fields.items()}
        return length

    def index_field(self, field_name, vocab):
        """use `vocab` to index certain field
        """
        self.indexes[field_name] = self.fields[field_name].index(vocab)

    def index_all(self, vocab):
        """use `vocab` to index all fields
        """
        if self.has_index:
            print("error")
            return self.indexes
        indexes = {name : field.index(vocab) for name, field in self.fields.items()}
        self.indexes = indexes
        return indexes

    def to_tensor(self, padding_length: dict):
        tensorX = {}
        tensorY = {}
        for name, field in self.fields.items():
            if field.is_target:
                tensorY[name] = field.to_tensor(padding_length[name])
            else:
                tensorX[name] = field.to_tensor(padding_length[name])
            
        return tensorX, tensorY



class Instance(object):
    """An instance which consists of Fields is an example in the DataSet.

    """

    def __init__(self, **fields):
        self.fields = fields

    def add_field(self, field_name, field):
        self.fields[field_name] = field
        return self

    def __getitem__(self, name):
        if name in self.fields:
            return self.fields[name]
        else:
            raise KeyError("{} not found".format(name))

    def __setitem__(self, name, field):
        return self.add_field(name, field)

    def __getattr__(self, item):
        if hasattr(self, 'fields') and item in self.fields:
            return self.fields[item]
        else:
            raise AttributeError('{} does not exist.'.format(item))

    def __setattr__(self, key, value):
        if hasattr(self, 'fields'):
            self.__setitem__(key, value)
        else:
            super().__setattr__(key, value)

    def __repr__(self):
        return self.fields.__repr__()

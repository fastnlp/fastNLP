

class Instance(object):
    """An instance which consists of Fields is an example in the DataSet.

    """

    def __init__(self, **fields):
        self.fields = fields

    def add_field(self, field_name, field):
        self.fields[field_name] = field
        return self

    def rename_field(self, old_name, new_name):
        if old_name in self.fields:
            self.fields[new_name] = self.fields.pop(old_name)
        else:
            raise KeyError("error, no such field: {}".format(old_name))
        return self

    def set_target(self, **fields):
        for name, val in fields.items():
            if name in self.fields:
                self.fields[name].is_target = val
        return self

    def __getitem__(self, name):
        if name in self.fields:
            return self.fields[name]
        else:
            raise KeyError("{} not found".format(name))

    def __setitem__(self, name, field):
        return self.add_field(name, field)

    def __repr__(self):
        return self.fields.__repr__()

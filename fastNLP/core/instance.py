class Instance(object):
    """An Instance is an example of data.
        Example::
            ins = Instance(field_1=[1, 1, 1], field_2=[2, 2, 2])
            ins["field_1"]
            >>[1, 1, 1]
            ins.add_field("field_3", [3, 3, 3])

        :param fields: a dict of (str: list).

    """

    def __init__(self, **fields):
        """

        :param fields: 可能是一维或者二维的 list or np.array
        """
        self.fields = fields

    def add_field(self, field_name, field):
        """Add a new field to the instance.

        :param field_name: str, the name of the field.
        """
        self.fields[field_name] = field

    def __getitem__(self, name):
        if name in self.fields:
            return self.fields[name]
        else:
            raise KeyError("{} not found".format(name))

    def __setitem__(self, name, field):
        return self.add_field(name, field)

    def __repr__(self):
        s = '\''
        return "{" + ",\n".join(
            "\'" + field_name + "\': " + str(self.fields[field_name]) + \
            f" type={(str(type(self.fields[field_name]))).split(s)[1]}" for field_name in self.fields) + "}"

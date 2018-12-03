class Instance(object):
    """An Instance is an example of data. It is the collection of Fields.

    ::
        Instance(field_1=[1, 1, 1], field_2=[2, 2, 2])

    """

    def __init__(self, **fields):
        """

        :param fields: a dict of (str: list).
        """
        self.fields = fields

    def add_field(self, field_name, field):
        """Add a new field to the instance.

        :param field_name: str, the name of the field.
        :param field:
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
        return "{" + ",\n".join(
            "\'" + field_name + "\': " + str(self.fields[field_name]) for field_name in self.fields) + "}"

"""
Instance文档

 .. _Instance:

Instance是fastNLP中对应于一个sample的类。一个sample可以认为是fastNLP中的一个Instance对象。一个具像化的表示类似与 DataSet_
出那个表中所展示的一行。

"""



class Instance(object):
    def __init__(self, **fields):
        """Instance的初始化如下面的Example所示

        Example::

            ins = Instance(field_1=[1, 1, 1], field_2=[2, 2, 2])
            ins["field_1"]
            >>[1, 1, 1]
            ins.add_field("field_3", [3, 3, 3])

            ins = Instance(**{'x1': 1, 'x2':np.zeros((3, 4))})
        """
        self.fields = fields

    def add_field(self, field_name, field):
        """向Instance中增加一个field

        :param str field_name: 新增field的名称
        :param Any field: 新增field的内容
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

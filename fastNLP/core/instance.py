"""
instance 模块实现了Instance 类在fastNLP中对应sample。一个sample可以认为是一个Instance类型的对象。
便于理解的例子可以参考文档 :doc:`fastNLP.core.dataset` 中的表格

"""
__all__ = [
    "Instance"
]


class Instance(object):
    """
    别名：:class:`fastNLP.Instance` :class:`fastNLP.core.instance.Instance`

    Instance是fastNLP中对应一个sample的类。每个sample在fastNLP中是一个Instance对象。
    Instance一般与 :class:`~fastNLP.DataSet` 一起使用, Instance的初始化如下面的Example所示::
    
        >>>from fastNLP import Instance
        >>>ins = Instance(field_1=[1, 1, 1], field_2=[2, 2, 2])
        >>>ins["field_1"]
        [1, 1, 1]
        >>>ins.add_field("field_3", [3, 3, 3])
        >>>ins = Instance(**{'x1': 1, 'x2':np.zeros((3, 4))})
    """
    
    def __init__(self, **fields):
        
        self.fields = fields
    
    def add_field(self, field_name, field):
        """
        向Instance中增加一个field

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

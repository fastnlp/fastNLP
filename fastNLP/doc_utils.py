r"""undocumented
用于辅助生成 fastNLP 文档的代码
"""

__all__ = []

import inspect
import sys


def doc_process(m):
    for name, obj in inspect.getmembers(m):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if obj.__module__ != m.__name__:
                if obj.__doc__ is None:
                    # print(name, obj.__doc__)
                    pass
                else:
                    module_name = obj.__module__
                    
                    # 识别并标注类和函数在不同层次中的位置
                    
                    while 1:
                        defined_m = sys.modules[module_name]
                        try:
                            if "undocumented" not in defined_m.__doc__ and name in defined_m.__all__:
                                obj.__doc__ = r"别名 :class:`" + m.__name__ + "." + name + "`" \
                                              + " :class:`" + module_name + "." + name + "`\n" + obj.__doc__
                                break
                            module_name = ".".join(module_name.split('.')[:-1])
                            if module_name == m.__name__:
                                # print(name, ": not found defined doc.")
                                break
                        except:
                            print("Warning: Module {} lacks `__doc__`".format(module_name))
                            break

                    # 识别并标注基类，只有基类也在 fastNLP 中定义才显示
                    
                    if inspect.isclass(obj):
                        for base in obj.__bases__:
                            if base.__module__.startswith("fastNLP"):
                                parts = base.__module__.split(".") + []
                                module_name, i = "fastNLP", 1
                                for i in range(len(parts) - 1):
                                    defined_m = sys.modules[module_name]
                                    try:
                                        if "undocumented" not in defined_m.__doc__ and name in defined_m.__all__:
                                            obj.__doc__ = r"基类 :class:`" + defined_m.__name__ + "." + base.__name__ + "` \n\n" + obj.__doc__
                                            break
                                        module_name += "." + parts[i + 1]
                                    except:
                                        print("Warning: Module {} lacks `__doc__`".format(module_name))
                                        break

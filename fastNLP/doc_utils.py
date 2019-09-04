import inspect
import sys


def doc_process(m):
    for name, obj in inspect.getmembers(m):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            if obj.__module__ != m.__name__:
                if obj.__doc__ is None:
                    print(name, obj.__doc__)
                else:
                    module_name = obj.__module__
                    while 1:
                        defined_m = sys.modules[module_name]
                        if "undocumented" not in defined_m.__doc__ and name in defined_m.__all__:
                            obj.__doc__ = r"定义在 :class:`" + module_name + "." + name + "`\n" + obj.__doc__
                            break
                        module_name = ".".join(module_name.split('.')[:-1])
                        if module_name == m.__name__:
                            print(name, ": not found defined doc.")
                            break

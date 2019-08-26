import os
import sys


def find_all_modules():
    modules = {}
    children = {}
    to_doc = set()
    root = '../fastNLP'
    for path, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.py'):
                name = ".".join(path.split('/')[1:])
                if file.split('.')[0] != "__init__":
                    name = name + '.' + file.split('.')[0]
                __import__(name)
                m = sys.modules[name]
                modules[name] = m
                try:
                    m.__all__
                except:
                    print(name, "__all__ missing")
                    continue
                if m.__doc__ is None:
                    print(name, "__doc__ missing")
                    continue
                if "undocumented" not in m.__doc__:
                    to_doc.add(name)
    for module in to_doc:
        t = ".".join(module.split('.')[:-1])
        if t in to_doc:
            if t not in children:
                children[t] = set()
            children[t].add(module)
    for m in children:
        children[m] = sorted(children[m])
    return modules, to_doc, children


def create_rst_file(modules, name, children):
    m = modules[name]
    with open("./source/" + name + ".rst", "w") as fout:
        t = "=" * len(name)
        fout.write(name + "\n")
        fout.write(t + "\n")
        fout.write("\n")
        fout.write(".. automodule:: " + name + "\n")
        if len(m.__all__) > 0:
            fout.write("   :members: " + ", ".join(m.__all__) + "\n")
            fout.write("   :inherited-members:\n")
        fout.write("\n")
        if name in children:
            fout.write("子模块\n------\n\n.. toctree::\n\n")
            for module in children[name]:
                fout.write("   " + module + "\n")


def main():
    modules, to_doc, children = find_all_modules()
    for name in to_doc:
        create_rst_file(modules, name, children)


if __name__ == "__main__":
    main()

import inspect
import os
import sys


def _colored_string(string: str, color: str or int) -> str:
    """在终端中显示一串有颜色的文字
    :param string: 在终端中显示的文字
    :param color: 文字的颜色
    :return:
    """
    if isinstance(color, str):
        color = {
            "black": 30, "Black": 30, "BLACK": 30,
            "red": 31, "Red": 31, "RED": 31,
            "green": 32, "Green": 32, "GREEN": 32,
            "yellow": 33, "Yellow": 33, "YELLOW": 33,
            "blue": 34, "Blue": 34, "BLUE": 34,
            "purple": 35, "Purple": 35, "PURPLE": 35,
            "cyan": 36, "Cyan": 36, "CYAN": 36,
            "white": 37, "White": 37, "WHITE": 37
        }[color]
    return "\033[%dm%s\033[0m" % (color, string)


def gr(string, flag):
    if flag:
        return _colored_string(string, "green")
    else:
        return _colored_string(string, "red")


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
        if name != "fastNLP.core" and len(m.__all__) > 0:
            fout.write("   :members: " + ", ".join(m.__all__) + "\n")
            short = name[len("fastNLP."):]
            if not (short.startswith('models') or short.startswith('modules') or short.startswith('embeddings')):
                fout.write("   :inherited-members:\n")
        fout.write("\n")
        if name in children:
            fout.write("子模块\n------\n\n.. toctree::\n   :maxdepth: 1\n\n")
            for module in children[name]:
                fout.write("   " + module + "\n")


def check_file(m, name):
    names = name.split('.')
    test_name = "test." + ".".join(names[1:-1]) + ".test_" + names[-1]
    try:
        __import__(test_name)
        tm = sys.modules[test_name]
    except ModuleNotFoundError:
        tm = None
    tested = tm is not None
    funcs = {}
    classes = {}
    for item, obj in inspect.getmembers(m):
        if inspect.isclass(obj) and obj.__module__ == name and not obj.__name__.startswith('_'):
            this = (obj.__doc__ is not None, tested and obj.__name__ in dir(tm), {})
            for i in dir(obj):
                func = getattr(obj, i)
                if inspect.isfunction(func) and not i.startswith('_'):
                    this[2][i] = (func.__doc__ is not None, False)
            classes[obj.__name__] = this
        if inspect.isfunction(obj) and obj.__module__ == name and not obj.__name__.startswith('_'):
            this = (obj.__doc__ is not None, tested and obj.__name__ in dir(tm))  # docs
            funcs[obj.__name__] = this
    return funcs, classes


def check_files(modules, out=None):
    for name in sorted(modules.keys()):
        print(name, file=out)
        funcs, classes = check_file(modules[name], name)
        if out is None:
            for f in funcs:
                print("%-30s \t %s \t %s" % (f, gr("文档", funcs[f][0]), gr("测试", funcs[f][1])))
            for c in classes:
                print("%-30s \t %s \t %s" % (c, gr("文档", classes[c][0]), gr("测试", classes[c][1])))
                methods = classes[c][2]
                for f in methods:
                    print("  %-28s \t %s" % (f, gr("文档", methods[f][0])))
        else:
            for f in funcs:
                if not funcs[f][0]:
                    print("缺少文档 %s" % (f), file=out)
                if not funcs[f][1]:
                    print("缺少测试 %s" % (f), file=out)
            for c in classes:
                if not classes[c][0]:
                    print("缺少文档 %s" % (c), file=out)
                if not classes[c][1]:
                    print("缺少测试 %s" % (c), file=out)
                methods = classes[c][2]
                for f in methods:
                    if not methods[f][0]:
                        print("缺少文档 %s" % (c + "." + f), file=out)
            print(file=out)


def main_check():
    sys.path.append("..")
    print(_colored_string('Getting modules...', "Blue"))
    modules, to_doc, children = find_all_modules()
    print(_colored_string('Done!', "Green"))
    print(_colored_string('Creating rst files...', "Blue"))
    for name in to_doc:
        create_rst_file(modules, name, children)
    print(_colored_string('Done!', "Green"))
    print(_colored_string('Checking all files...', "Blue"))
    check_files(modules, out=open("results.txt", "w"))
    print(_colored_string('Done!', "Green"))


def check_file_r(file_path):
    with open(file_path) as fin:
        content = fin.read()
    index = -3
    cuts = []
    while index != -1:
        index = content.find('"""',index+3)
        cuts.append(index)
    cuts = cuts[:-1]
    assert len(cuts)%2 == 0
    write_content = ""
    last = 0
    for i in range(len(cuts)//2):
        start, end = cuts[i+i], cuts[i+i+1]
        if content[start-1] == "r":
            write_content += content[last:end+3]
        else:
            write_content += content[last:start] + "r"
            write_content += content[start:end+3]
        last = end + 3
    write_content += content[last:]
    with open(file_path, "w") as fout:
        fout.write(write_content)


def add_r(base_path='../fastNLP'):
    for path, _, files in os.walk(base_path):
        for f in files:
            if f.endswith(".py"):
                check_file_r(os.path.abspath(os.path.join(path,f)))
                sys.exit(0)


if __name__ == "__main__":
    add_r()

import os


def find_all(path='../fastNLP'):
    head_list = []
    alias_list = []
    for path, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.py'):
                name = ".".join(path.split('/')[1:])
                if file.split('.')[0] != "__init__":
                    name = name + '.' + file.split('.')[0]
                if len(name.split('.')) < 3 or name.startswith('fastNLP.core'):
                    heads, alias = find_one(path + '/' + file)
                    for h in heads:
                        head_list.append(name + "." + h)
                    for a in alias:
                        alias_list.append(a)
    heads = {}
    for h in head_list:
        end = h.split('.')[-1]
        file = h[:-len(end) - 1]
        if end not in heads:
            heads[end] = set()
        heads[end].add(file)
    alias = {}
    for a in alias_list:
        for each in a:
            end = each.split('.')[-1]
            file = each[:-len(end) - 1]
            if end not in alias:
                alias[end] = set()
            alias[end].add(file)
    print("IN alias NOT IN heads")
    for item in alias:
        if item not in heads:
            print(item, alias[item])
        elif len(heads[item]) != 2:
            print(item, alias[item], heads[item])
    
    print("\n\nIN heads NOT IN alias")
    for item in heads:
        if item not in alias:
            print(item, heads[item])


def find_class(path):
    with open(path, 'r') as fin:
        lines = fin.readlines()
    pars = {}
    for i, line in enumerate(lines):
        if line.strip().startswith('class'):
            line = line.strip()[len('class'):-1].strip()
            if line[-1] == ')':
                line = line[:-1].split('(')
                name = line[0].strip()
                parents = line[1].split(',')
                for i in range(len(parents)):
                    parents[i] = parents[i].strip()
                if len(parents) == 1:
                    pars[name] = parents[0]
                else:
                    pars[name] = tuple(parents)
    return pars


def find_one(path):
    head_list = []
    alias = []
    with open(path, 'r') as fin:
        lines = fin.readlines()
    flag = False
    for i, line in enumerate(lines):
        if line.strip().startswith('__all__'):
            line = line.strip()[len('__all__'):].strip()
            if line[-1] == ']':
                line = line[1:-1].strip()[1:].strip()
                head_list.append(line.strip("\"").strip("\'").strip())
            else:
                flag = True
        elif line.strip() == ']':
            flag = False
        elif flag:
            line = line.strip()[:-1].strip("\"").strip("\'").strip()
            if len(line) == 0 or line[0] == '#':
                continue
            head_list.append(line)
        if line.startswith('def') or line.startswith('class'):
            if lines[i + 2].strip().startswith("别名："):
                names = lines[i + 2].strip()[len("别名："):].split()
                names[0] = names[0][len(":class:`"):-1]
                names[1] = names[1][len(":class:`"):-1]
                alias.append((names[0], names[1]))
    return head_list, alias


if __name__ == "__main__":
    find_all()  # use to check __all__

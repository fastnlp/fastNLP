import os


def shorten(file, to_delete, cut=False):
    if file.endswith("index.rst") or file.endswith("conf.py"):
        return
    res = []
    with open(file, "r") as fin:
        lines = fin.readlines()
    for line in lines:
        if cut and line.rstrip() == "Submodules":
            break
        else:
            res.append(line.rstrip())
    for i, line in enumerate(res):
        if line.endswith(" package"):
            res[i] = res[i][:-len(" package")]
            res[i + 1] = res[i + 1][:-len(" package")]
        elif line.endswith(" module"):
            res[i] = res[i][:-len(" module")]
            res[i + 1] = res[i + 1][:-len(" module")]
        else:
            for name in to_delete:
                if line.endswith(name):
                    res[i] = "del"

    with open(file, "w") as fout:
        for line in res:
            if line != "del":
                print(line, file=fout)


def clear(path='./source/'):
    files = os.listdir(path)
    to_delete = [
        "fastNLP.core.dist_trainer",
        "fastNLP.core.predictor",

        "fastNLP.io.file_reader",
        "fastNLP.io.config_io",

        "fastNLP.embeddings.contextual_embedding",

        "fastNLP.modules.dropout",
        "fastNLP.models.base_model",
        "fastNLP.models.bert",
        "fastNLP.models.enas_utils",
        "fastNLP.models.enas_controller",
        "fastNLP.models.enas_model",
        "fastNLP.models.enas_trainer",
    ]
    for file in files:
        if not os.path.isdir(path + file):
            res = file.split('.')
            if len(res) > 4:
                to_delete.append(file[:-4])
            elif len(res) == 4:
                shorten(path + file, to_delete, True)
            else:
                shorten(path + file, to_delete)
    for file in to_delete:
        try:
            os.remove(path + file + ".rst")
        except:
            pass


clear()

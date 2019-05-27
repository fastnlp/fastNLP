from fastNLP import DataSet


def read_PWKP(path):
    """
    :param path:
    :return: an object of DataSet
    读入PWKP数据的函数，输入PWKP（https://nlpprogress.com/english/simplification.html）数据的地址，返回fastnlp中的dataset对象
    """
    with open(path, 'r', encoding='utf8') as f:
        tmp_sentences = []
        _dict = {}
        _dict['origin'] = []
        _dict['simplified'] = []
        for line_idx, line in enumerate(f):
            line = line.strip()
            if len(line) == 0:
                if len(tmp_sentences) <= 1:
                    print("Line {} : Not good example! No original sentence or simplified sentence! ".format(line_idx))
                else:
                    _dict['origin'].append(tmp_sentences[0])
                    _dict['simplified'].append(tmp_sentences[1:])
                tmp_sentences = []
            else:
                tmp_sentences.append(line)
    return DataSet(_dict)


if __name__ == 'main':
    dataset = read_PWKP('PWKP_108016')
    print(dataset)

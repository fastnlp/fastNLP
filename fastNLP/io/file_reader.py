"""
此模块用于给其它模块提供读取文件的函数，没有为用户提供 API
"""
import json


def _read_csv(path, encoding='utf-8', headers=None, sep=',', dropna=True):
    """
    Construct a generator to read csv items.

    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param headers: file's headers, if None, make file's first line as headers. default: None
    :param sep: separator for each column. default: ','
    :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, csv item)
    """
    with open(path, 'r', encoding=encoding) as f:
        start_idx = 0
        if headers is None:
            headers = f.readline().rstrip('\r\n')
            headers = headers.split(sep)
            start_idx += 1
        elif not isinstance(headers, (list, tuple)):
                raise TypeError("headers should be list or tuple, not {}." \
                        .format(type(headers)))
        for line_idx, line in enumerate(f, start_idx):
            contents = line.rstrip('\r\n').split(sep)
            if len(contents) != len(headers):
                if dropna:
                    continue
                else:
                    raise ValueError("Line {} has {} parts, while header has {} parts." \
                                     .format(line_idx, len(contents), len(headers)))
            _dict = {}
            for header, content in zip(headers, contents):
                _dict[header] = content
            yield line_idx, _dict


def _read_json(path, encoding='utf-8', fields=None, dropna=True):
    """
    Construct a generator to read json items.

    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param fields: json object's fields that needed, if None, all fields are needed. default: None
    :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, json item)
    """
    if fields:
        fields = set(fields)
    with open(path, 'r', encoding=encoding) as f:
        for line_idx, line in enumerate(f):
            data = json.loads(line)
            if fields is None:
                yield line_idx, data
                continue
            _res = {}
            for k, v in data.items():
                if k in fields:
                    _res[k] = v
            if len(_res) < len(fields):
                if dropna:
                    continue
                else:
                    raise ValueError('invalid instance at line: {}'.format(line_idx))
            yield line_idx, _res


def _read_conll(path, encoding='utf-8', indexes=None, dropna=True):
    """
    Construct a generator to read conll items.

    :param path: file path
    :param encoding: file's encoding, default: utf-8
    :param indexes: conll object's column indexes that needed, if None, all columns are needed. default: None
    :param dropna: weather to ignore and drop invalid data,
            :if False, raise ValueError when reading invalid data. default: True
    :return: generator, every time yield (line number, conll item)
    """
    def parse_conll(sample):
        sample = list(map(list, zip(*sample)))
        sample = [sample[i] for i in indexes]
        for f in sample:
            if len(f) <= 0:
                raise ValueError('empty field')
        return sample
    with open(path, 'r', encoding=encoding) as f:
        sample = []
        start = next(f)
        if '-DOCSTART-' not in start:
            sample.append(start.split())
        for line_idx, line in enumerate(f, 1):
            if line.startswith('\n'):
                if len(sample):
                    try:
                        res = parse_conll(sample)
                        sample = []
                        yield line_idx, res
                    except Exception as e:
                        if dropna:
                            continue
                        raise ValueError('invalid instance at line: {}'.format(line_idx))
            elif line.startswith('#'):
                continue
            else:
                sample.append(line.split())
        if len(sample) > 0:
            try:
                res = parse_conll(sample)
                yield line_idx, res
            except Exception as e:
                if dropna:
                    return
                raise ValueError('invalid instance at line: {}'.format(line_idx))

class ConllxDataLoader(object):
    def load(self, path):
        datalist = []
        with open(path, 'r', encoding='utf-8') as f:
            sample = []
            for line in f:
                if line.startswith('\n'):
                    datalist.append(sample)
                    sample = []
                elif line.startswith('#'):
                    continue
                else:
                    sample.append(line.split('\t'))
            if len(sample) > 0:
                datalist.append(sample)

        data = [self.get_one(sample) for sample in datalist]
        return list(filter(lambda x: x is not None, data))

    def get_one(self, sample):
        sample = list(map(list, zip(*sample)))
        if len(sample) == 0:
            return None
        for w in sample[7]:
            if w == '_':
                print('Error Sample {}'.format(sample))
                return None
        # return word_seq, pos_seq, head_seq, head_tag_seq
        return sample[1], sample[3], list(map(int, sample[6])), sample[7]


class MyDataloader:
    def load(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        data = self.parse(lines)
        return data

    def parse(self, lines):
        """
            [
                [word], [pos], [head_index], [head_tag]
            ]
        """
        sample = []
        data = []
        for i, line in enumerate(lines):
            line = line.strip()
            if len(line) == 0 or i + 1 == len(lines):
                data.append(list(map(list, zip(*sample))))
                sample = []
            else:
                sample.append(line.split())
        if len(sample) > 0:
            data.append(list(map(list, zip(*sample))))
        return data


def add_seg_tag(data):
    """

    :param data: list of ([word], [pos], [heads], [head_tags])
    :return: list of ([word], [pos])
    """

    _processed = []
    for word_list, pos_list, _, _ in data:
        new_sample = []
        for word, pos in zip(word_list, pos_list):
            if len(word) == 1:
                new_sample.append((word, 'S-' + pos))
            else:
                new_sample.append((word[0], 'B-' + pos))
                for c in word[1:-1]:
                    new_sample.append((c, 'M-' + pos))
                new_sample.append((word[-1], 'E-' + pos))
        _processed.append(list(map(list, zip(*new_sample))))
    return _processed
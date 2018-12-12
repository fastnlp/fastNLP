
from fastNLP.core.dataset import DataSet
from fastNLP.core.instance import Instance

def cut_long_sentence(sent, max_sample_length=200):
    sent_no_space = sent.replace(' ', '')
    cutted_sentence = []
    if len(sent_no_space) > max_sample_length:
        parts = sent.strip().split()
        new_line = ''
        length = 0
        for part in parts:
            length += len(part)
            new_line += part + ' '
            if length > max_sample_length:
                new_line = new_line[:-1]
                cutted_sentence.append(new_line)
                length = 0
                new_line = ''
        if new_line != '':
            cutted_sentence.append(new_line[:-1])
    else:
        cutted_sentence.append(sent)
    return cutted_sentence


class ConlluPOSReader(object):
    # 返回的Dataset包含words(list of list, 里层的list是character), tag两个field(list of str, str是标有BMES的tag)。
    def __init__(self):
        pass

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

        ds = DataSet()
        for sample in datalist:
            # print(sample)
            res = self.get_one(sample)
            if res is None:
                continue
            char_seq = []
            pos_seq = []
            for word, tag in zip(res[0], res[1]):
                if len(word)==1:
                    char_seq.append(word)
                    pos_seq.append('S-{}'.format(tag))
                elif len(word)>1:
                    pos_seq.append('B-{}'.format(tag))
                    for _ in range(len(word)-2):
                        pos_seq.append('M-{}'.format(tag))
                    pos_seq.append('E-{}'.format(tag))
                    char_seq.extend(list(word))
                else:
                    raise ValueError("Zero length of word detected.")

            ds.append(Instance(words=char_seq,
                               tag=pos_seq))

        return ds

    def get_one(self, sample):
        if len(sample)==0:
            return None
        text = []
        pos_tags = []
        for w in sample:
            t1, t2, t3, t4 = w[1], w[3], w[6], w[7]
            if t3 == '_':
                return None
            text.append(t1)
            pos_tags.append(t2)
        return text, pos_tags

if __name__ == '__main__':
    reader = ConlluPOSReader()
    d = reader.load('/home/hyan/train.conllx')
    print('reader')
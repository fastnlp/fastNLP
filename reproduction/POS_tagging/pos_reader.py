from fastNLP.io.dataset_loader import ZhConllPOSReader


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


if __name__ == '__main__':
    reader = ZhConllPOSReader()
    d = reader.load('/home/hyan/train.conllx')
    print(d)
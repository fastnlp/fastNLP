
from collections import Counter

from fastNLP.api.processor import Processor
from fastNLP.core.dataset import DataSet


class CombineWordAndPosProcessor(Processor):
    def __init__(self, word_field_name, pos_field_name):
        super(CombineWordAndPosProcessor, self).__init__(None, None)

        self.word_field_name = word_field_name
        self.pos_field_name = pos_field_name

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))

        for ins in dataset:
            chars = ins[self.word_field_name]
            bmes_pos = ins[self.pos_field_name]
            word_list = []
            pos_list = []
            pos_stack_cnt = Counter()
            char_stack = []
            for char, p in zip(chars, bmes_pos):
                parts = p.split('-')
                pre = parts[0]
                post = parts[1]
                if pre.lower() == 's':
                    if len(pos_stack_cnt) != 0:
                        pos = pos_stack_cnt.most_common(1)[0][0]
                        pos_list.append(pos)
                        word_list.append(''.join(char_stack))
                    pos_list.append(post)
                    word_list.append(char)
                    char_stack.clear()
                    pos_stack_cnt.clear()
                elif pre.lower() == 'e':
                    pos_stack_cnt.update([post])
                    char_stack.append(char)
                    pos = pos_stack_cnt.most_common(1)[0][0]
                    pos_list.append(pos)
                    word_list.append(''.join(char_stack))
                    char_stack.clear()
                    pos_stack_cnt.clear()
                elif pre.lower() == 'b':
                    if len(pos_stack_cnt) != 0:
                        pos = pos_stack_cnt.most_common(1)[0][0]
                        pos_list.append(pos)
                        word_list.append(''.join(char_stack))
                    char_stack.clear()
                    pos_stack_cnt.clear()
                    char_stack.append(char)
                    pos_stack_cnt.update([post])
                else:
                    char_stack.append(char)
                    pos_stack_cnt.update([post])

            ins['word_list'] = word_list
            ins['pos_list'] = pos_list

        return dataset


class PosOutputStrProcessor(Processor):
    def __init__(self, word_field_name, pos_field_name):
        super(PosOutputStrProcessor, self).__init__(None, None)

        self.word_field_name = word_field_name
        self.pos_field_name = pos_field_name
        self.sep = '_'

    def process(self, dataset):
        assert isinstance(dataset, DataSet), "Only Dataset class is allowed, not {}.".format(type(dataset))

        for ins in dataset:
            word_list = ins[self.word_field_name]
            pos_list = ins[self.pos_field_name]

            word_pos_list = []
            for word, pos in zip(word_list, pos_list):
                word_pos_list.append(word + self.sep + pos)
            #TODO 应该可以定制
            ins['word_pos_output'] = '  '.join(word_pos_list)

        return dataset


if __name__ == '__main__':
    chars = ['迈', '向', '充', '满', '希', '望', '的', '新', '世', '纪', '—', '—', '一', '九', '九', '八', '年', '新', '年', '讲', '话', '（', '附', '图', '片', '１', '张', '）']
    bmes_pos = ['B-v', 'E-v', 'B-v', 'E-v', 'B-n', 'E-n', 'S-u', 'S-a', 'B-n', 'E-n', 'B-w', 'E-w', 'B-t', 'M-t', 'M-t', 'M-t', 'E-t', 'B-t', 'E-t', 'B-n', 'E-n', 'S-w', 'S-v', 'B-n', 'E-n', 'S-m', 'S-q', 'S-w']


    word_list = []
    pos_list = []
    pos_stack_cnt = Counter()
    char_stack = []
    for char, p in zip(''.join(chars), bmes_pos):
        parts = p.split('-')
        pre = parts[0]
        post = parts[1]
        if pre.lower() == 's':
            if len(pos_stack_cnt) != 0:
                pos = pos_stack_cnt.most_common(1)[0][0]
                pos_list.append(pos)
                word_list.append(''.join(char_stack))
            pos_list.append(post)
            word_list.append(char)
            char_stack.clear()
            pos_stack_cnt.clear()
        elif pre.lower() == 'e':
            pos_stack_cnt.update([post])
            char_stack.append(char)
            pos = pos_stack_cnt.most_common(1)[0][0]
            pos_list.append(pos)
            word_list.append(''.join(char_stack))
            char_stack.clear()
            pos_stack_cnt.clear()
        elif pre.lower() == 'b':
            if len(pos_stack_cnt) != 0:
                pos = pos_stack_cnt.most_common(1)[0][0]
                pos_list.append(pos)
                word_list.append(''.join(char_stack))
            char_stack.clear()
            pos_stack_cnt.clear()
            char_stack.append(char)
            pos_stack_cnt.update([post])
        else:
            char_stack.append(char)
            pos_stack_cnt.update([post])

    print(word_list)
    print(pos_list)

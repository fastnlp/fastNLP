
import re


class SpanConverter:
    def __init__(self, replace_tag, pattern):
        super(SpanConverter, self).__init__()

        self.replace_tag = replace_tag
        self.pattern = pattern

    def find_certain_span_and_replace(self, sentence):
        replaced_sentence = ''
        prev_end = 0
        for match in re.finditer(self.pattern, sentence):
            start, end = match.span()
            span = sentence[start:end]
            replaced_sentence += sentence[prev_end:start] + \
                self.span_to_special_tag(span)
            prev_end = end
        replaced_sentence += sentence[prev_end:]

        return replaced_sentence

    def span_to_special_tag(self, span):

        return self.replace_tag

    def find_certain_span(self, sentence):
        spans = []
        for match in re.finditer(self.pattern, sentence):
            spans.append(match.span())
        return spans


class AlphaSpanConverter(SpanConverter):
    def __init__(self):
        replace_tag = '<ALPHA>'
        # 理想状态下仅处理纯为字母的情况, 但不处理<[a-zA-Z]+>(因为这应该是特殊的tag).
        pattern = '[a-zA-Z]+(?=[\u4e00-\u9fff ,%.!<\\-"])'

        super(AlphaSpanConverter, self).__init__(replace_tag, pattern)


class DigitSpanConverter(SpanConverter):
    def __init__(self):
        replace_tag = '<NUM>'
        pattern = '\d[\d\\.]*(?=[\u4e00-\u9fff  ,%.!<-])'

        super(DigitSpanConverter, self).__init__(replace_tag, pattern)

    def span_to_special_tag(self, span):
        # return self.special_tag
        if span[0] == '0' and len(span) > 2:
            return '<NUM>'
        decimal_point_count = 0  # one might have more than one decimal pointers
        for idx, char in enumerate(span):
            if char == '.' or char == '﹒' or char == '·':
                decimal_point_count += 1
        if span[-1] == '.' or span[-1] == '﹒' or span[
            -1] == '·':  # last digit being decimal point means this is not a number
            if decimal_point_count == 1:
                return span
            else:
                return '<UNKDGT>'
        if decimal_point_count == 1:
            return '<DEC>'
        elif decimal_point_count > 1:
            return '<UNKDGT>'
        else:
            return '<NUM>'


class TimeConverter(SpanConverter):
    def __init__(self):
        replace_tag = '<TOC>'
        pattern = '\d+[:：∶][\d:：∶]+(?=[\u4e00-\u9fff  ,%.!<-])'

        super().__init__(replace_tag, pattern)



class MixNumAlphaConverter(SpanConverter):
    def __init__(self):
        replace_tag = '<MIX>'
        pattern = None

        super().__init__(replace_tag, pattern)

    def find_certain_span_and_replace(self, sentence):
        replaced_sentence = ''
        start = 0
        matching_flag = False
        number_flag = False
        alpha_flag = False
        link_flag = False
        slash_flag = False
        bracket_flag = False
        for idx in range(len(sentence)):
            if re.match('[0-9a-zA-Z/\\(\\)\'′&\\-]', sentence[idx]):
                if not matching_flag:
                    replaced_sentence += sentence[start:idx]
                    start = idx
                if re.match('[0-9]', sentence[idx]):
                    number_flag = True
                elif re.match('[\'′&\\-]', sentence[idx]):
                    link_flag = True
                elif re.match('/', sentence[idx]):
                    slash_flag = True
                elif re.match('[\\(\\)]', sentence[idx]):
                    bracket_flag = True
                else:
                    alpha_flag = True
                matching_flag = True
            elif re.match('[\\.]', sentence[idx]):
                pass
            else:
                if matching_flag:
                    if (number_flag and alpha_flag) or (link_flag and alpha_flag) \
                            or (slash_flag and alpha_flag) or (link_flag and number_flag) \
                            or (number_flag and bracket_flag) or (bracket_flag and alpha_flag):
                        span = sentence[start:idx]
                        start = idx
                        replaced_sentence += self.span_to_special_tag(span)
                    matching_flag = False
                    number_flag = False
                    alpha_flag = False
                    link_flag = False
                    slash_flag = False
                    bracket_flag = False

        replaced_sentence += sentence[start:]
        return replaced_sentence

    def find_certain_span(self, sentence):
        spans = []
        start = 0
        matching_flag = False
        number_flag = False
        alpha_flag = False
        link_flag = False
        slash_flag = False
        bracket_flag = False
        for idx in range(len(sentence)):
            if re.match('[0-9a-zA-Z/\\(\\)\'′&\\-]', sentence[idx]):
                if not matching_flag:
                    start = idx
                if re.match('[0-9]', sentence[idx]):
                    number_flag = True
                elif re.match('[\'′&\\-]', sentence[idx]):
                    link_flag = True
                elif re.match('/', sentence[idx]):
                    slash_flag = True
                elif re.match('[\\(\\)]', sentence[idx]):
                    bracket_flag = True
                else:
                    alpha_flag = True
                matching_flag = True
            elif re.match('[\\.]', sentence[idx]):
                pass
            else:
                if matching_flag:
                    if (number_flag and alpha_flag) or (link_flag and alpha_flag) \
                            or (slash_flag and alpha_flag) or (link_flag and number_flag) \
                            or (number_flag and bracket_flag) or (bracket_flag and alpha_flag):
                        spans.append((start, idx))
                        start = idx

                    matching_flag = False
                    number_flag = False
                    alpha_flag = False
                    link_flag = False
                    slash_flag = False
                    bracket_flag = False

        return spans



class EmailConverter(SpanConverter):
    def __init__(self):
        replaced_tag = "<EML>"
        pattern = '[0-9a-zA-Z]+[@][.﹒0-9a-zA-Z@]+(?=[\u4e00-\u9fff  ,%.!<\\-"$])'

        super(EmailConverter, self).__init__(replaced_tag, pattern)
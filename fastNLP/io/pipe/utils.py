from typing import List
from ...core.vocabulary import Vocabulary
from ...core.const import Const

def iob2(tags:List[str])->List[str]:
    """
    检查数据是否是合法的IOB数据，如果是IOB1会被自动转换为IOB2。两种格式的区别见https://datascience.stackexchange.com/questions/37824/difference-between-iob-and-iob2-format

    :param tags: 需要转换的tags
    """
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        split = tag.split("-")
        if len(split) != 2 or split[0] not in ["I", "B"]:
            raise TypeError("The encoding schema is not a valid IOB type.")
        if split[0] == "B":
            continue
        elif i == 0 or tags[i - 1] == "O":  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = "B" + tag[1:]
    return tags

def iob2bioes(tags:List[str])->List[str]:
    """
    将iob的tag转换为bioes编码
    :param tags:
    :return:
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        else:
            split = tag.split('-')[0]
            if split == 'B':
                if i+1!=len(tags) and tags[i+1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('B-', 'S-'))
            elif split == 'I':
                if i + 1<len(tags) and tags[i+1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('I-', 'E-'))
            else:
                raise TypeError("Invalid IOB format.")
    return new_tags


def get_tokenizer(tokenizer:str, lang='en'):
    """

    :param str tokenizer: 获取tokenzier方法
    :param str lang: 语言，当前仅支持en
    :return: 返回tokenize函数
    """
    if tokenizer == 'spacy':
        import spacy
        spacy.prefer_gpu()
        if lang != 'en':
            raise RuntimeError("Spacy only supports en right right.")
        en = spacy.load(lang)
        tokenizer = lambda x: [w.text for w in en.tokenizer(x)]
    elif tokenizer == 'raw':
        tokenizer = _raw_split
    else:
        raise RuntimeError("Only support `spacy`, `raw` tokenizer.")
    return tokenizer


def _raw_split(sent):
    return sent.split()


def _indexize(data_bundle, input_field_name=Const.INPUT, target_field_name=Const.TARGET):
    """
    在dataset中的field_name列建立词表，Const.TARGET列建立词表，并把词表加入到data_bundle中。

    :param data_bundle:
    :param: str input_field_name:
    :param: str target_field_name: 这一列的vocabulary没有unknown和padding
    :return:
    """
    src_vocab = Vocabulary()
    src_vocab.from_dataset(data_bundle.datasets['train'], field_name=input_field_name,
                           no_create_entry_dataset=[dataset for name, dataset in data_bundle.datasets.items() if
                                                    name != 'train'])
    src_vocab.index_dataset(*data_bundle.datasets.values(), field_name=input_field_name)

    tgt_vocab = Vocabulary(unknown=None, padding=None)
    tgt_vocab.from_dataset(data_bundle.datasets['train'], field_name=target_field_name)
    tgt_vocab.index_dataset(*data_bundle.datasets.values(), field_name=target_field_name)

    data_bundle.set_vocab(src_vocab, input_field_name)
    data_bundle.set_vocab(tgt_vocab, target_field_name)

    return data_bundle


def _add_words_field(data_bundle, lower=False):
    """
    给data_bundle中的dataset中复制一列words. 并根据lower参数判断是否需要小写化

    :param data_bundle:
    :param bool lower:是否要小写化
    :return: 传入的DataBundle
    """
    data_bundle.copy_field(field_name=Const.RAW_WORD, new_field_name=Const.INPUT, ignore_miss_dataset=True)

    if lower:
        for name, dataset in data_bundle.datasets.items():
            dataset[Const.INPUT].lower()
    return data_bundle


def _add_chars_field(data_bundle, lower=False):
    """
    给data_bundle中的dataset中复制一列chars. 并根据lower参数判断是否需要小写化

    :param data_bundle:
    :param bool lower:是否要小写化
    :return: 传入的DataBundle
    """
    data_bundle.copy_field(field_name=Const.RAW_CHAR, new_field_name=Const.CHAR_INPUT, ignore_miss_dataset=True)

    if lower:
        for name, dataset in data_bundle.datasets.items():
            dataset[Const.CHAR_INPUT].lower()
    return data_bundle


def _drop_empty_instance(data_bundle, field_name):
    """
    删除data_bundle的DataSet中存在的某个field为空的情况

    :param data_bundle: DataBundle
    :param str field_name: 对哪个field进行检查，如果为None，则任意field为空都会删掉
    :return: 传入的DataBundle
    """
    def empty_instance(ins):
        if field_name:
            field_value = ins[field_name]
            if field_value in ((), {}, [], ''):
                return True
            return False
        for _, field_value in ins.items():
            if field_value in ((), {}, [], ''):
                return True
        return False

    for name, dataset in data_bundle.datasets.items():
        dataset.drop(empty_instance)

    return data_bundle



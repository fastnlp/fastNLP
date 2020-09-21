r"""undocumented"""

__all__ = [
    "iob2",
    "iob2bioes",
    "get_tokenizer",
]

from typing import List
import warnings

from ...core.const import Const
from ...core.vocabulary import Vocabulary
from ...core._logger import logger


def iob2(tags: List[str]) -> List[str]:
    r"""
    检查数据是否是合法的IOB数据，如果是IOB1会被自动转换为IOB2。两种格式的区别见
    https://datascience.stackexchange.com/questions/37824/difference-between-iob-and-iob2-format

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


def iob2bioes(tags: List[str]) -> List[str]:
    r"""
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
                if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('B-', 'S-'))
            elif split == 'I':
                if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('I-', 'E-'))
            else:
                raise TypeError("Invalid IOB format.")
    return new_tags


def get_tokenizer(tokenize_method: str, lang='en'):
    r"""

    :param str tokenize_method: 获取tokenzier方法
    :param str lang: 语言，当前仅支持en
    :return: 返回tokenize函数
    """
    tokenizer_dict = {
        'spacy': None,
        'raw': _raw_split,
        'cn-char': _cn_char_split,
    }
    if tokenize_method == 'spacy':
        import spacy
        spacy.prefer_gpu()
        if lang != 'en':
            raise RuntimeError("Spacy only supports en right right.")
        en = spacy.load(lang)
        tokenizer = lambda x: [w.text for w in en.tokenizer(x)]
    elif tokenize_method in tokenizer_dict:
        tokenizer = tokenizer_dict[tokenize_method]
    else:
        raise RuntimeError(f"Only support {tokenizer_dict.keys()} tokenizer.")
    return tokenizer


def _cn_char_split(sent):
    return [chars for chars in sent]


def _raw_split(sent):
    return sent.split()


def _indexize(data_bundle, input_field_names=Const.INPUT, target_field_names=Const.TARGET):
    r"""
    在dataset中的field_name列建立词表，Const.TARGET列建立词表，并把词表加入到data_bundle中。

    :param ~fastNLP.DataBundle data_bundle:
    :param: str,list input_field_names:
    :param: str,list target_field_names: 这一列的vocabulary没有unknown和padding
    :return:
    """
    if isinstance(input_field_names, str):
        input_field_names = [input_field_names]
    if isinstance(target_field_names, str):
        target_field_names = [target_field_names]
    for input_field_name in input_field_names:
        src_vocab = Vocabulary()
        src_vocab.from_dataset(*[ds for name, ds in data_bundle.iter_datasets() if 'train' in name],
                               field_name=input_field_name,
                               no_create_entry_dataset=[ds for name, ds in data_bundle.iter_datasets()
                                                        if ('train' not in name) and (ds.has_field(input_field_name))]
                               )
        src_vocab.index_dataset(*data_bundle.datasets.values(), field_name=input_field_name)
        data_bundle.set_vocab(src_vocab, input_field_name)
    
    for target_field_name in target_field_names:
        tgt_vocab = Vocabulary(unknown=None, padding=None)
        tgt_vocab.from_dataset(*[ds for name, ds in data_bundle.iter_datasets() if 'train' in name],
                               field_name=target_field_name,
                               no_create_entry_dataset=[ds for name, ds in data_bundle.iter_datasets()
                                                        if ('train' not in name) and (ds.has_field(target_field_name))]
                               )
        if len(tgt_vocab._no_create_word) > 0:
            warn_msg = f"There are {len(tgt_vocab._no_create_word)} `{target_field_name}` labels" \
                       f" in {[name for name in data_bundle.datasets.keys() if 'train' not in name]} " \
                       f"data set but not in train data set!.\n" \
                       f"These label(s) are {tgt_vocab._no_create_word}"
            warnings.warn(warn_msg)
            logger.warning(warn_msg)
        tgt_vocab.index_dataset(*[ds for ds in data_bundle.datasets.values() if ds.has_field(target_field_name)], field_name=target_field_name)
        data_bundle.set_vocab(tgt_vocab, target_field_name)
    
    return data_bundle


def _add_words_field(data_bundle, lower=False):
    r"""
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
    r"""
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
    r"""
    删除data_bundle的DataSet中存在的某个field为空的情况

    :param ~fastNLP.DataBundle data_bundle:
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


def _granularize(data_bundle, tag_map):
    r"""
    该函数对data_bundle中'target'列中的内容进行转换。

    :param data_bundle:
    :param dict tag_map: 将target列中的tag做以下的映射，比如{"0":0, "1":0, "3":1, "4":1}, 则会删除target为"2"的instance，
        且将"1"认为是第0类。
    :return: 传入的data_bundle
    """
    if tag_map is None:
        return data_bundle
    for name in list(data_bundle.datasets.keys()):
        dataset = data_bundle.get_dataset(name)
        dataset.apply_field(lambda target: tag_map.get(target, -100), field_name=Const.TARGET,
                            new_field_name=Const.TARGET)
        dataset.drop(lambda ins: ins[Const.TARGET] == -100)
        data_bundle.set_dataset(dataset, name)
    return data_bundle

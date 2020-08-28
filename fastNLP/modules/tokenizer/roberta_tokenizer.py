r"""

"""

__all__ = [
    "RobertaTokenizer"
]

import json
from .gpt2_tokenizer import GPT2Tokenizer
from fastNLP.io.file_utils import _get_file_name_base_on_postfix
from ...io.file_utils import _get_roberta_dir

PRETRAINED_ROBERTA_POSITIONAL_EMBEDDINGS_SIZES = {
    "roberta-base": 512,
    "roberta-large": 512,
    "roberta-large-mnli": 512,
    "distilroberta-base": 512,
    "roberta-base-openai-detector": 512,
    "roberta-large-openai-detector": 512,
}


class RobertaTokenizer(GPT2Tokenizer):

    vocab_files_names = {
        "vocab_file": "vocab.json",
        "merges_file": "merges.txt",
    }

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
        **kwargs
    ):
        super().__init__(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors=errors,
            bos_token=bos_token,
            eos_token=eos_token,
            unk_token=unk_token,
            sep_token=sep_token,
            cls_token=cls_token,
            pad_token=pad_token,
            mask_token=mask_token,
            **kwargs,
        )
        self.max_len_single_sentence = self.max_len - 2  # take into account special tokens
        self.max_len_sentences_pair = self.max_len - 4  # take into account special tokens

    @classmethod
    def from_pretrained(cls, model_dir_or_name, *inputs, **kwargs):
        """

        :param str model_dir_or_name: 目录或者缩写名
        :param kwargs:
        :return:
        """
        # 它需要两个文件，第一个是vocab.json，第二个是merge_file?
        model_dir = _get_roberta_dir(model_dir_or_name)
        # 里面会包含四个文件vocab.json, merge.txt, config.json, model.bin

        tokenizer_config_file = _get_file_name_base_on_postfix(model_dir, 'config.json')
        with open(tokenizer_config_file, encoding="utf-8") as tokenizer_config_handle:
            init_kwargs = json.load(tokenizer_config_handle)
        # Set max length if needed
        if model_dir_or_name in PRETRAINED_ROBERTA_POSITIONAL_EMBEDDINGS_SIZES:
            # if we're using a pretrained model, ensure the tokenizer
            # wont index sequences longer than the number of positional embeddings
            max_len = PRETRAINED_ROBERTA_POSITIONAL_EMBEDDINGS_SIZES[model_dir_or_name]
            if max_len is not None and isinstance(max_len, (int, float)):
                init_kwargs["max_len"] = min(init_kwargs.get("max_len", int(1e12)), max_len)

        # 将vocab, merge加入到init_kwargs中
        if 'vocab_file' in kwargs:  # 如果指定了词表则用指定词表
            init_kwargs['vocab_file'] = kwargs['vocab_file']
        else:
            init_kwargs['vocab_file'] = _get_file_name_base_on_postfix(model_dir, RobertaTokenizer.vocab_files_names['vocab_file'])
        init_kwargs['merges_file'] = _get_file_name_base_on_postfix(model_dir, RobertaTokenizer.vocab_files_names['merges_file'])

        init_inputs = init_kwargs.pop("init_inputs", ())
        # Instantiate tokenizer.
        try:
            tokenizer = cls(*init_inputs, **init_kwargs)
        except OSError:
            OSError(
                "Unable to load vocabulary from file. "
                "Please check that the provided vocabulary is accessible and not corrupted."
            )

        return tokenizer


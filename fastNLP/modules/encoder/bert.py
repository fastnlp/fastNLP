
import os
from torch import nn
import torch
from ...core import Vocabulary
from ...io.file_utils import _get_base_url, cached_path
from ._bert import _WordPieceBertModel


class BertWordPieceEncoder(nn.Module):
    """
    可以通过读取vocabulary使用的Bert的Encoder。传入vocab，然后调用index_datasets方法在vocabulary中生成word piece的表示。

    :param fastNLP.Vocabulary vocab: 词表
    :param str model_dir_or_name: 模型所在目录或者模型的名称。默认值为``en-base-uncased``
    :param str layers:最终结果中的表示。以','隔开层数，可以以负数去索引倒数几层
    :param bool requires_grad: 是否需要gradient。
    """
    def __init__(self, vocab:Vocabulary, model_dir_or_name:str='en-base', layers:str='-1',
                 requires_grad:bool=False):
        super().__init__()
        PRETRAIN_URL = _get_base_url('bert')
        PRETRAINED_BERT_MODEL_DIR = {'en': 'bert-base-cased-f89bfe08.zip',
                                     'en-base-uncased': 'bert-base-uncased-3413b23c.zip',
                                     'en-base-cased': 'bert-base-cased-f89bfe08.zip',
                                     'en-large-uncased': 'bert-large-uncased-20939f45.zip',
                                     'en-large-cased': 'bert-large-cased-e0cf90fc.zip',

                                     'cn': 'bert-base-chinese-29d0a84a.zip',
                                     'cn-base': 'bert-base-chinese-29d0a84a.zip',

                                     'multilingual': 'bert-base-multilingual-cased-1bd364ee.zip',
                                     'multilingual-base-uncased': 'bert-base-multilingual-uncased-f8730fe4.zip',
                                     'multilingual-base-cased': 'bert-base-multilingual-cased-1bd364ee.zip',
                                     }

        if model_dir_or_name in PRETRAINED_BERT_MODEL_DIR:
            model_name = PRETRAINED_BERT_MODEL_DIR[model_dir_or_name]
            model_url = PRETRAIN_URL + model_name
            model_dir = cached_path(model_url)
            # 检查是否存在
        elif os.path.isdir(model_dir_or_name):
            model_dir = model_dir_or_name
        else:
            raise ValueError(f"Cannot recognize {model_dir_or_name}.")

        self.model = _WordPieceBertModel(model_dir=model_dir, vocab=vocab, layers=layers)
        self._embed_size = len(self.model.layers) * self.model.encoder.hidden_size
        self.requires_grad = requires_grad

    @property
    def requires_grad(self):
        """
        Embedding的参数是否允许优化。True: 所有参数运行优化; False: 所有参数不允许优化; None: 部分允许优化、部分不允许
        :return:
        """
        requires_grads = set([param.requires_grad for name, param in self.named_parameters()])
        if len(requires_grads)==1:
            return requires_grads.pop()
        else:
            return None

    @requires_grad.setter
    def requires_grad(self, value):
        for name, param in self.named_parameters():
            param.requires_grad = value

    @property
    def embed_size(self):
        return self._embed_size

    def index_datasets(self, *datasets):
        """
        根据datasets中的'words'列对datasets进行word piece的index。

        Example::

        :param datasets:
        :return:
        """
        self.model.index_dataset(*datasets)

    def forward(self, words, token_type_ids=None):
        """
        计算words的bert embedding表示。计算之前会在每句话的开始增加[CLS]在结束增加[SEP], 并根据include_cls_sep判断要不要
            删除这两个表示。

        :param words: batch_size x max_len
        :param token_type_ids: batch_size x max_len, 用于区分前一句和后一句话
        :return: torch.FloatTensor. batch_size x max_len x (768*len(self.layers))
        """
        outputs = self.model(words, token_type_ids)
        outputs = torch.cat([*outputs], dim=-1)

        return outputs
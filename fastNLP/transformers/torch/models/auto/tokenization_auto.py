# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Auto Tokenizer class. """

from collections import OrderedDict
from typing import TYPE_CHECKING, Optional, Tuple

from ...file_utils import (
    is_sentencepiece_available,
    is_tokenizers_available,
)

if TYPE_CHECKING:
    # This significantly improves completion suggestion performance when
    # the transformers package is used with Microsoft's Pylance language server.
    TOKENIZER_MAPPING_NAMES: OrderedDict[str, Tuple[Optional[str], Optional[str]]] = OrderedDict()
else:
    TOKENIZER_MAPPING_NAMES = OrderedDict(
        [
            ("fnet", ("FNetTokenizer", "FNetTokenizerFast" if is_tokenizers_available() else None)),
            ("retribert", ("RetriBertTokenizer", "RetriBertTokenizerFast" if is_tokenizers_available() else None)),
            ("roformer", ("RoFormerTokenizer", "RoFormerTokenizerFast" if is_tokenizers_available() else None)),
            (
                "t5",
                (
                    "T5Tokenizer" if is_sentencepiece_available() else None,
                    "T5TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "mt5",
                (
                    "MT5Tokenizer" if is_sentencepiece_available() else None,
                    "MT5TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("mobilebert", ("MobileBertTokenizer", "MobileBertTokenizerFast" if is_tokenizers_available() else None)),
            ("distilbert", ("DistilBertTokenizer", "DistilBertTokenizerFast" if is_tokenizers_available() else None)),
            (
                "albert",
                (
                    "AlbertTokenizer" if is_sentencepiece_available() else None,
                    "AlbertTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "camembert",
                (
                    "CamembertTokenizer" if is_sentencepiece_available() else None,
                    "CamembertTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "pegasus",
                (
                    "PegasusTokenizer" if is_sentencepiece_available() else None,
                    "PegasusTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "mbart",
                (
                    "MBartTokenizer" if is_sentencepiece_available() else None,
                    "MBartTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "xlm-roberta",
                (
                    "XLMRobertaTokenizer" if is_sentencepiece_available() else None,
                    "XLMRobertaTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("marian", ("MarianTokenizer" if is_sentencepiece_available() else None, None)),
            ("blenderbot-small", ("BlenderbotSmallTokenizer", None)),
            ("blenderbot", ("BlenderbotTokenizer", None)),
            ("bart", ("BartTokenizer", "BartTokenizerFast")),
            ("longformer", ("LongformerTokenizer", "LongformerTokenizerFast" if is_tokenizers_available() else None)),
            ("roberta", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
            (
                "reformer",
                (
                    "ReformerTokenizer" if is_sentencepiece_available() else None,
                    "ReformerTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("electra", ("ElectraTokenizer", "ElectraTokenizerFast" if is_tokenizers_available() else None)),
            ("funnel", ("FunnelTokenizer", "FunnelTokenizerFast" if is_tokenizers_available() else None)),
            ("lxmert", ("LxmertTokenizer", "LxmertTokenizerFast" if is_tokenizers_available() else None)),
            ("layoutlm", ("LayoutLMTokenizer", "LayoutLMTokenizerFast" if is_tokenizers_available() else None)),
            ("layoutlmv2", ("LayoutLMv2Tokenizer", "LayoutLMv2TokenizerFast" if is_tokenizers_available() else None)),
            (
                "dpr",
                (
                    "DPRQuestionEncoderTokenizer",
                    "DPRQuestionEncoderTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "squeezebert",
                ("SqueezeBertTokenizer", "SqueezeBertTokenizerFast" if is_tokenizers_available() else None),
            ),
            ("bert", ("BertTokenizer", "BertTokenizerFast" if is_tokenizers_available() else None)),
            ("openai-gpt", ("OpenAIGPTTokenizer", "OpenAIGPTTokenizerFast" if is_tokenizers_available() else None)),
            ("gpt2", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            ("transfo-xl", ("TransfoXLTokenizer", None)),
            (
                "xlnet",
                (
                    "XLNetTokenizer" if is_sentencepiece_available() else None,
                    "XLNetTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("flaubert", ("FlaubertTokenizer", None)),
            ("xlm", ("XLMTokenizer", None)),
            ("ctrl", ("CTRLTokenizer", None)),
            ("fsmt", ("FSMTTokenizer", None)),
            ("bert-generation", ("BertGenerationTokenizer" if is_sentencepiece_available() else None, None)),
            ("deberta", ("DebertaTokenizer", "DebertaTokenizerFast" if is_tokenizers_available() else None)),
            ("deberta-v2", ("DebertaV2Tokenizer" if is_sentencepiece_available() else None, None)),
            ("rag", ("RagTokenizer", None)),
            ("xlm-prophetnet", ("XLMProphetNetTokenizer" if is_sentencepiece_available() else None, None)),
            ("speech_to_text", ("Speech2TextTokenizer" if is_sentencepiece_available() else None, None)),
            ("speech_to_text_2", ("Speech2Text2Tokenizer", None)),
            ("m2m_100", ("M2M100Tokenizer" if is_sentencepiece_available() else None, None)),
            ("prophetnet", ("ProphetNetTokenizer", None)),
            ("mpnet", ("MPNetTokenizer", "MPNetTokenizerFast" if is_tokenizers_available() else None)),
            ("tapas", ("TapasTokenizer", None)),
            ("led", ("LEDTokenizer", "LEDTokenizerFast" if is_tokenizers_available() else None)),
            ("convbert", ("ConvBertTokenizer", "ConvBertTokenizerFast" if is_tokenizers_available() else None)),
            (
                "big_bird",
                (
                    "BigBirdTokenizer" if is_sentencepiece_available() else None,
                    "BigBirdTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("ibert", ("RobertaTokenizer", "RobertaTokenizerFast" if is_tokenizers_available() else None)),
            ("wav2vec2", ("Wav2Vec2CTCTokenizer", None)),
            ("hubert", ("Wav2Vec2CTCTokenizer", None)),
            ("gpt_neo", ("GPT2Tokenizer", "GPT2TokenizerFast" if is_tokenizers_available() else None)),
            ("luke", ("LukeTokenizer", None)),
            ("bigbird_pegasus", ("PegasusTokenizer", "PegasusTokenizerFast" if is_tokenizers_available() else None)),
            ("canine", ("CanineTokenizer", None)),
            ("bertweet", ("BertweetTokenizer", None)),
            ("bert-japanese", ("BertJapaneseTokenizer", None)),
            ("splinter", ("SplinterTokenizer", "SplinterTokenizerFast")),
            ("byt5", ("ByT5Tokenizer", None)),
            (
                "cpm",
                (
                    "CpmTokenizer" if is_sentencepiece_available() else None,
                    "CpmTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            ("herbert", ("HerbertTokenizer", "HerbertTokenizerFast" if is_tokenizers_available() else None)),
            ("phobert", ("PhobertTokenizer", None)),
            (
                "barthez",
                (
                    "BarthezTokenizer" if is_sentencepiece_available() else None,
                    "BarthezTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "mbart50",
                (
                    "MBart50Tokenizer" if is_sentencepiece_available() else None,
                    "MBart50TokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "rembert",
                (
                    "RemBertTokenizer" if is_sentencepiece_available() else None,
                    "RemBertTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
            (
                "clip",
                (
                    "CLIPTokenizer",
                    "CLIPTokenizerFast" if is_tokenizers_available() else None,
                ),
            ),
        ]
    )
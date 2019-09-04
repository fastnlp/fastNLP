from fastNLP.io.dataset_loader import JsonLoader,DataSet,Instance
from fastNLP.io.file_reader import _read_json
from fastNLP.core.vocabulary import Vocabulary
from fastNLP.io.data_bundle import DataBundle
from reproduction.coreference_resolution.model.config import Config
import reproduction.coreference_resolution.model.preprocess as preprocess


class CRLoader(JsonLoader):
    def __init__(self, fields=None, dropna=False):
        super().__init__(fields, dropna)

    def _load(self, path):
        """
        加载数据
        :param path:
        :return:
        """
        dataset = DataSet()
        for idx, d in _read_json(path, fields=self.fields_list, dropna=self.dropna):
            if self.fields:
                ins = {self.fields[k]: v for k, v in d.items()}
            else:
                ins = d
            dataset.append(Instance(**ins))
        return dataset

    def process(self, paths, **kwargs):
        data_info = DataBundle()
        for name in ['train', 'test', 'dev']:
            data_info.datasets[name] = self.load(paths[name])

        config = Config()
        vocab = Vocabulary().from_dataset(*data_info.datasets.values(), field_name='sentences')
        vocab.build_vocab()
        word2id = vocab.word2idx

        char_dict = preprocess.get_char_dict(config.char_path)
        data_info.vocabs = vocab

        genres = {g: i for i, g in enumerate(["bc", "bn", "mz", "nw", "pt", "tc", "wb"])}

        for name, ds in data_info.datasets.items():
            ds.apply(lambda x: preprocess.doc2numpy(x['sentences'], word2id, char_dict, max(config.filter),
                                                        config.max_sentences, is_train=name=='train')[0],
                         new_field_name='doc_np')
            ds.apply(lambda x: preprocess.doc2numpy(x['sentences'], word2id, char_dict, max(config.filter),
                                                        config.max_sentences, is_train=name=='train')[1],
                         new_field_name='char_index')
            ds.apply(lambda x: preprocess.doc2numpy(x['sentences'], word2id, char_dict, max(config.filter),
                                                        config.max_sentences, is_train=name=='train')[2],
                         new_field_name='seq_len')
            ds.apply(lambda x: preprocess.speaker2numpy(x["speakers"], config.max_sentences, is_train=name=='train'),
                         new_field_name='speaker_ids_np')
            ds.apply(lambda x: genres[x["doc_key"][:2]], new_field_name='genre')

            ds.set_ignore_type('clusters')
            ds.set_padder('clusters', None)
            ds.set_input("sentences", "doc_np", "speaker_ids_np", "genre", "char_index", "seq_len")
            ds.set_target("clusters")

        # train_dev, test = self.ds.split(348 / (2802 + 343 + 348), shuffle=False)
        # train, dev = train_dev.split(343 / (2802 + 343), shuffle=False)

        return data_info




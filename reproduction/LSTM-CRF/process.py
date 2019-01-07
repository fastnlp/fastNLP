from fastNLP.io.dataset_loader import Conll2003Loader

def lower_case(dataset_list, key):
    for dataset in dataset_list:
        dataset.apply(lambda x: \
          list(map(lambda item: item.lower(), x[key])), new_field_name=key)

def build_vocab(dataset_list, key):
    vocab = Vocabulary(min_freq=1)
    for dataset in dataset_list:
        dataset.apply(lambda x: [vocab.add(word) for word in x[key]])
    vocab.build_vocab()
    return vocab

def build_index(dataset_list, key, index_key, vocab):
    for dataset in dataset_list:
        dataset.apply(lambda x: [vocab.to_index(word) for word in x[key]], new_field_name=index_key)

def load_data(data_loader, paths):
    datasets = []
    for path in paths:
        datasets.append(dataloader(path))
    return datasets
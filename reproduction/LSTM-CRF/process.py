from fastNLP.io.dataset_loader import Conll2003Loader
from fastNLP import Vocabulary


def lower_case(dataset_list, key):
    """
        Lower case each word in the given key on given datasets
    
        :param dataset_list: List of Dataset
        :param key: string for key
    """
    for dataset in dataset_list:
        dataset.apply(lambda x: \
          list(map(lambda item: item.lower(), x[key])), new_field_name=key)

def build_vocab(dataset_list, key):
    """
        Build vocab from the given datasets on certain key
    
        :param dataset_list: List of Dataset
        :param key: string for key
        :return vocab: Vocabulary, the vocab created
    """
    vocab = Vocabulary(min_freq=1)
    for dataset in dataset_list:
        dataset.apply(lambda x: [vocab.add(word) for word in x[key]])
    vocab.build_vocab()
    return vocab

def build_index(dataset_list, key, index_key, vocab):
    """
        Build word_2_index column on dataset based on the vocab
    
        :param dataset_list: List of Dataset
        :param key: string for key
        :param index_key: string, the new column name
        :param vocab: Vocabulary, the given vocab
    """
    for dataset in dataset_list:
        dataset.apply(lambda x: [vocab.to_index(word) for word in x[key]], new_field_name=index_key)
        
def build_origin_len(dataset_list, key, origin_len_key):
    """
        Build origin len column for dataset
    
        :param dataset_list: List of Dataset
        :param key: string for key
        :param index_key: string, the new column name
    """
    for dataset in dataset_list:
        dataset.apply(lambda x: len(x[key]), new_field_name=origin_len_key)
    
    

def load_data(data_loader, paths):
    """
        Load the data based on certain paths
        
        :param data_loader: the loader to load the data
        :param paths: the list of file path to load the data
        :return datasets: the list of loaded datasets
    """
    datasets = []
    for path in paths:
        datasets.append(data_loader.load(path))
    return datasets
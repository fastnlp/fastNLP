import _pickle
import os


# the first vocab in dict with the index = 5

def save_pickle(obj, pickle_path, file_name):
    """Save an object into a pickle file.

    :param obj: an object
    :param pickle_path: str, the directory where the pickle file is to be saved
    :param file_name: str, the name of the pickle file. In general, it should be ended by "pkl".
    """
    if not os.path.exists(pickle_path):
        os.mkdir(pickle_path)
        print("make dir {} before saving pickle file".format(pickle_path))
    with open(os.path.join(pickle_path, file_name), "wb") as f:
        _pickle.dump(obj, f)
    print("{} saved in {}".format(file_name, pickle_path))


def load_pickle(pickle_path, file_name):
    """Load an object from a given pickle file.

    :param pickle_path: str, the directory where the pickle file is.
    :param file_name: str, the name of the pickle file.
    :return obj: an object stored in the pickle
    """
    with open(os.path.join(pickle_path, file_name), "rb") as f:
        obj = _pickle.load(f)
    print("{} loaded from {}".format(file_name, pickle_path))
    return obj


def pickle_exist(pickle_path, pickle_name):
    """Check if a given pickle file exists in the directory.

    :param pickle_path: the directory of target pickle file
    :param pickle_name: the filename of target pickle file
    :return: True if file exists else False
    """
    if not os.path.exists(pickle_path):
        os.makedirs(pickle_path)
    file_name = os.path.join(pickle_path, pickle_name)
    if os.path.exists(file_name):
        return True
    else:
        return False

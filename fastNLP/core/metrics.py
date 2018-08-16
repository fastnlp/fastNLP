import warnings
import numpy as np
import torch


def _conver_numpy(x):
    """
    convert input data to numpy array
    """
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.numpy()
    elif isinstance(x, list):
        return np.array(x)
    raise TypeError('cannot accept object: {}'.format(x))


def _check_same_len(*arrays, axis=0):
    """
    check if input array list has same length for one dimension
    """
    lens = set([x.shape[axis] for x in arrays if x is not None])
    return len(lens) == 1


def _label_types(y):
    """
    determine the type
    "binary"
    "multiclass"
    "multiclass-multioutput"
    "multilabel"
    "unknown"
    """
    # never squeeze the first dimension
    y = y.squeeze() if y.shape[0] > 1 else y.resize(1, -1)
    shape = y.shape
    if len(shape) < 1:
        raise ValueError('cannot accept data: {}'.format(y))
    if len(shape) == 1:
        return 'multiclass' if np.unique(y).shape[0] > 2 else 'binary', y
    if len(shape) == 2:
        return 'multiclass-multioutput' if np.unique(y).shape[0] > 2 else 'multilabel', y
    return 'unknown', y


def _check_data(y_true, y_pred):
    """
    check if y_true and y_pred is same type of data e.g both binary or multiclass
    """
    y_true, y_pred = _conver_numpy(y_true), _conver_numpy(y_pred)
    if not _check_same_len(y_true, y_pred):
        raise ValueError('cannot accept data with different shape {0}, {1}'.format(y_true, y_pred))
    type_true, y_true = _label_types(y_true)
    type_pred, y_pred = _label_types(y_pred)

    type_set = set(['binary', 'multiclass'])
    if type_true in type_set and type_pred in type_set:
        return type_true if type_true == type_pred else 'multiclass', y_true, y_pred

    type_set = set(['multiclass-multioutput', 'multilabel'])
    if type_true in type_set and type_pred in type_set:
        return type_true if type_true == type_pred else 'multiclass-multioutput', y_true, y_pred

    raise ValueError('cannot accept data mixed of {0} and {1} target'.format(type_true, type_pred))


def _weight_sum(y, normalize=True, sample_weight=None):
    if normalize:
        return np.average(y, weights=sample_weight)
    if sample_weight is None:
        return y.sum()
    else:
        return np.dot(y, sample_weight)


def accuracy_score(y_true, y_pred, normalize=True, sample_weight=None):
    y_type, y_true, y_pred = _check_data(y_true, y_pred)
    if y_type == 'multiclass-multioutput':
        raise ValueError('cannot accept data type {0}'.format(y_type))
    if y_type == 'multilabel':
        equel = (y_true == y_pred).sum(1)
        count = equel == y_true.shape[1]
    else:
        count = y_true == y_pred
    return _weight_sum(count, normalize=normalize, sample_weight=sample_weight)


def recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary'):
    y_type, y_true, y_pred = _check_data(y_true, y_pred)
    if average == 'binary':
        if y_type != 'binary':
            raise ValueError("data type is {} but  use average type {}".format(y_type, average))
        else:
            pos = (y_true == pos_label)
            tp = np.logical_and((y_true == y_pred), pos).sum()
            pos_sum = pos.sum()
            return tp / pos_sum if pos_sum > 0 else 0
    elif average == None:
        y_labels = set(list(np.unique(y_true)))
        if labels is None:
            labels = list(y_labels)
        else:
            for i in labels:
                if (i not in y_labels and y_type != 'multilabel') or (y_type == 'multilabel' and i >= y_true.shape[1]):
                    warnings.warn('label {} is not contained in data'.format(i), UserWarning)

        if y_type in ['binary', 'multiclass']:
            y_pred_right = y_true == y_pred
            pos_list = [y_true == i for i in labels]
            pos_sum_list = [pos_i.sum() for pos_i in pos_list]
            return np.array([np.logical_and(y_pred_right, pos_i).sum() / sum_i if sum_i > 0 else 0 \
                             for pos_i, sum_i in zip(pos_list, pos_sum_list)])
        elif y_type == 'multilabel':
            y_pred_right = y_true == y_pred
            pos = (y_true == pos_label)
            tp = np.logical_and(y_pred_right, pos).sum(0)
            pos_sum = pos.sum(0)
            return np.array([tp[i] / pos_sum[i] if pos_sum[i] > 0 else 0 for i in labels])
        else:
            raise ValueError('not support targets type {}'.format(y_type))
    raise ValueError('not support for average type {}'.format(average))


def precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary'):
    y_type, y_true, y_pred = _check_data(y_true, y_pred)
    if average == 'binary':
        if y_type != 'binary':
            raise ValueError("data type is {} but  use average type {}".format(y_type, average))
        else:
            pos = (y_true == pos_label)
            tp = np.logical_and((y_true == y_pred), pos).sum()
            pos_pred = (y_pred == pos_label).sum()
            return tp / pos_pred if pos_pred > 0 else 0
    elif average == None:
        y_labels = set(list(np.unique(y_true)))
        if labels is None:
            labels = list(y_labels)
        else:
            for i in labels:
                if (i not in y_labels and y_type != 'multilabel') or (y_type == 'multilabel' and i >= y_true.shape[1]):
                    warnings.warn('label {} is not contained in data'.format(i), UserWarning)

        if y_type in ['binary', 'multiclass']:
            y_pred_right = y_true == y_pred
            pos_list = [y_true == i for i in labels]
            pos_sum_list = [(y_pred == i).sum() for i in labels]
            return np.array([np.logical_and(y_pred_right, pos_i).sum() / sum_i if sum_i > 0 else 0 \
                             for pos_i, sum_i in zip(pos_list, pos_sum_list)])
        elif y_type == 'multilabel':
            y_pred_right = y_true == y_pred
            pos = (y_true == pos_label)
            tp = np.logical_and(y_pred_right, pos).sum(0)
            pos_sum = (y_pred == pos_label).sum(0)
            return np.array([tp[i] / pos_sum[i] if pos_sum[i] > 0 else 0 for i in labels])
        else:
            raise ValueError('not support targets type {}'.format(y_type))
    raise ValueError('not support for average type {}'.format(average))


def f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary'):
    precision = precision_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average)
    recall = recall_score(y_true, y_pred, labels=labels, pos_label=pos_label, average=average)
    if isinstance(precision, np.ndarray):
        res = 2 * precision * recall / (precision + recall)
        res[(precision + recall) <= 0] = 0
        return res
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0


def classification_report(y_true, y_pred, labels=None, target_names=None, digits=2):
    raise NotImplementedError


if __name__ == '__main__':
    y = np.array([1, 0, 1, 0, 1, 1])
    print(_label_types(y))

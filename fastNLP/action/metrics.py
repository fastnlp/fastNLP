"""
To do:
 设计评判结果的各种指标。如果涉及向量，使用numpy。
 参考http://scikit-learn.org/stable/modules/classes.html#classification-metrics
 建议是每种metric写成一个函数 （由Tester的evaluate函数调用）
 参数表里只需考虑基本的参数即可，可以没有像它那么多的参数配置
  
    support numpy array and torch tensor
"""
import numpy as np
import torch
import sklearn.metrics as M


def _conver_numpy(x):
    '''
    converte input data to numpy array
    '''
    if isinstance(x, np.ndarray): 
        return x
    elif isinstance(x, torch.Tensor): 
        return x.numpy()
    elif isinstance(x, list): 
        return np.array(x)
    raise TypeError('cannot accept obejct: {}'.format(x))

def _check_same_len(*arrays, axis=0):
    '''
    check if input array list has same length for one dimension
    '''
    lens = set([x.shape[axis] for x in arrays if x is not None])
    return len(lens) == 1
        

def _label_types(y):
    '''
    determine the type
    "binary"
    "multiclass"
    "multiclass-multioutput"
    "multilabel"
    '''
    # never squeeze the first dimension
    y = np.squeeze(y, list(range(1, len(y.shape))))
    shape = y.shape
    if len(shape) < 1: 
        raise ValueError('cannot accept data: {}'.format(y))
    if len(shape) == 1:
        return 'multiclass' if np.unique(y).shape[0] > 2 else 'binary', y
    if len(shape) == 2:
        return 'multiclass-multioutput' if np.unique(y).shape[0] > 2 else 'multilabel', y
    return 'unknown', y
        

def _check_data(y_true, y_pred):
    '''
    check if y_true and y_pred is same type of data e.g both binary or multiclass
    '''
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


def recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None):
    raise NotImplementedError

def precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None):
    raise NotImplementedError

def f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None):
    raise NotImplementedError

def classification_report(y_true, y_pred, labels=None, target_names=None, sample_weight=None, digits=2):
    raise NotImplementedError

if __name__ == '__main__':
    y = np.array([1,0,1,0,1,1])
    print(_label_types(y))
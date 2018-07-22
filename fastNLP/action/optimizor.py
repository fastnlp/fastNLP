from torch import optim

def get_torch_optimizor(params, alg_name='sgd', **args):
    '''
    construct pytorch optimizor by algorithm's name
    optimizor's argurments can be splicified, for different optimizor's argurments, please see pytorch doc

    usage:
        optimizor = get_torch_optimizor(model.parameters(), 'SGD', lr=0.01)

    '''
    
    name = alg_name.lower()
    if name == 'adadelta':
        return optim.Adadelta(params, **args)
    elif name == 'adagrad':
        return optim.Adagrad(params, **args)
    elif name == 'adam':
        return optim.Adam(params, **args)
    elif name == 'adamax':
        return optim.Adamax(params, **args)
    elif name == 'asgd':
        return optim.ASGD(params, **args)
    elif name == 'lbfgs':
        return optim.LBFGS(params, **args)
    elif name == 'rmsprop':
        return optim.RMSprop(params, **args)
    elif name == 'rprop':
        return optim.Rprop(params, **args)
    elif name == 'sgd':
        #SGD's parameter lr is required
        if 'lr' not in args:
            args['lr'] = 0.01
        return optim.SGD(params, **args)
    elif name == 'sparseadam':
        return optim.SparseAdam(params, **args)
    else:
        raise TypeError('no such optimizor named {}'.format(alg_name))


# example usage
if __name__ == '__main__':
    from torch.nn.modules import Linear
    net = Linear(2, 5)

    test1 = get_torch_optimizor(net.parameters(),'adam', lr=1e-2, weight_decay=1e-3)
    print(test1)
    test2 = get_torch_optimizor(net.parameters(), 'SGD')
    print(test2)
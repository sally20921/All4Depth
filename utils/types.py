import yacs
import numpy as np
import torch

def is_numpy(data):
    '''
    check if data is a numpy array
    '''
    return isinstance(data, np.ndarray)

def is_tensor(data):
    '''
    check if data is a torch tensor
    '''
    return type(data) == torch.Tensor

def is_tuple(data):
    '''
    check if data is a tuple
    '''
    return isinstance(data, tuple)

def is_list(data):
    '''
    check if data is a list
    '''
    return isinstance(data, list)

def is_dict(data):
    '''
    check if data is a dictionary
    '''
    return isinstance(data, dict)

def is_str(data):
    '''
    check if data is a string
    '''
    return isinstance(data, str)

def is_int(data):
    '''
    check if data is an integer
    '''
    return isinstance(data, int)

def is_seq(data):
    '''
    check if data is a list or tuple
    '''
    return is_tuple(data) or is_list(data)

def is_cfg(data):
    '''
    check if data is a configuration node
    '''
    return type(data) == yacs.config.CfgNode

from utils.types import is_list, is_int

def filter_dict(dictionary, keywords):
    '''
    returns only the keywords that are part of a dictionary

    Arguments:
        dictionary: dict
            dictionary for filtering
        keywords: list or str
            keywords that will be filtered

    returns:
        keywords: list or str
            list containing the keywords that are keys in dictionary
    '''
    return [key for key in keywords if key in dictionary]

def make_list(var, n=None):
    '''
    wraps the input into a list, and optionally repeats it to be size n

    Arguments:
        var: any
            variable to be wrapped in a list
        n: int
            how much the wrapped variable will be repeated

    returns:
        var_list: list
            list generated from var
    '''
    var = var if is_list(var) else [var]
    if n is None:
        return var
    else:
        assert len(var) == 1 or len(var) == n 'wrong list length for make_List'
        return var * n if len(var) == 1 else var


def same_shape(shape1, shape2):
    '''
    check if two shapes are the same
    
    Arguments:
        shape1: tuple
            first shape
        shape2: tuple
            second shape

    returns:
        flag: bool
            True if both shapes are the same (same length and dimensions)
    '''
    if len(shape1) != len(shape2):
        return False
    for i in range(len(shape1)):
        if shape1[i] != shape2[i]:
            return False

    return True


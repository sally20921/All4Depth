from utils.types import is_list, is_int

def filter_dict(dictionary, keywords):
    '''
    returns only the keywords (as a list) that are part of a dictionary

    Parameter
    ________
    dictionary: dict
        dictionary for filtering
    keywords: list or str
        keywords that will be filtered

    Returns
    ________
    keywords: list
        list containing the keywords that are keys in dictionary
    '''
    return [key for key in keywords if key in dictionary]
    # if keywords is a string, key is a list of characters that are in dictionary as key # var is a single value => a list containing single value 


def make_list(var, n=None): #optionally var can be a list, if that's the case, do nothing # if var is somehow not single value but has elements n, then do nothing even though there's a number in n

    # only multiply by n if var is a single value
    # var == single value or var == n values
    '''
    wrap the input (var) into a list, and optionally repeat it to be size n

    Parameter
    _________
    var: Any 
        Variable to be wrapped in a list
    n: int
        how much the wrapped variable will be repeated

    Returns
    _______
    var_list: list
        list generated from var
    
    '''
    var = var if is_list(var) else [var]
    if n is None:
        return var
    else:
        assert len(var) == 1 or len(var) == n, 'Wrong list length for make_list in utils.misc')
        return var * n if len(var) == 1 else var

def same_shape(shape1, shape2):
    '''
    checks if two tuples' shapes have exactly same elements in a tuple
    Parameter
    _________
    shape1: tuple
        first shape
    shape2: tuple
        second shape

    Returns
    _________
    flag: bool
        True if both shapes are the same (same length and dimension)
    '''
    if len(shape1) != len(shape2):
        return False
    for i, elem in enumerate(shape1):
        if elem != shape2[i]:
            return False
    return True

def parse_crop_borders(borders, shape):
    '''
    calculate the borders (parse the borders)  => for cropping.

    Parameters
    __________
    borders: tuple
        border input for parsing. can be one of the following forms:
        (int, int, int, int): y, height, x, width
        (int, int): y, x 
        --> in this case: height = image_height-y, width = image_width-x

    shape: tuple
        image_shape: (image_height, image_width)
            used to determine crop boundaries

    Returns
    _______         (x1, y1, x2, y2)
    borders: tuple (left, top, right, bottom)
        parsed borders for cropping
    '''
    # if borders is an empty tuple (no borders to crop), return the full image
    if len(borders) == 0:
        return 0, 0, shape[1], shape[0]

    # copy borders as a list for modification
    borders = list(borders).copy()
    # copy() method returns a new list. 

    # if the borders are 4-dimensional 
    if len(borders) == 4:
        borders = [borders[2], borders[0], borders[3], borders[1]]
                # put it in (x1, y1, width, height) order




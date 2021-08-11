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

# used in datasets.transforms
'''
torchvision.transforms.CenterCrop(size)
- crop the image at the center
- image is expecte to have [.... H, W] shape
- if image size is smaller than output size along any edge, image is padded with 0 and then center

def _get_image_size(img: Any) -> List[int]: # Tensor (.., H, W)
    # returns (w, h) of tensor image
    if isinstance(img, Image.Image):
        return img.size
    return [img.shape[-1], img.shape[-2]]

def center_crop(img: Tensor, output_size: List[int]) -> Tensor:
    image_width, image_height = _get_image_size(img)
    crop_height, crop_width = output_size

def pad(img: Tensor, padding: List[int], fill: int = 0, padding_mode: str="constant") -> Tensor:
    # padding: (left, top, right, bottom)
'''


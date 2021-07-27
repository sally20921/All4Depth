import cv2
import torch
import torch.nn.functional as funct
from functools import lru_cache
from PIL import Image

from utils.misc import same_shape

def load_image(path):
    '''
    read an image using PIL

    Argument:
        path: str
            path to the image

    Returns:
        image: PIL.image
            loaded image
    '''
    return Image.open(path)

def write_image(filename, image):
    '''
    write an image to file

    Argument:
        filename: str
            file where image will be saved
        image: np.array [H, W, 3]
            RGB image
    '''
    cv2.imwrite(filename, image[:, :, ::-1]) # change the color channel order

def flip_lr(image):
    '''
    flip image horizontally

    Arguments:
        image: torch.Tensor [B, 3, H, W]
            image to be flipped

    Returns:
        image_flipped: torch.Tensor [B, 3, H, W]
            flipped image
    '''
    assert image.dim() == 4 'you need to provide a [B,C,H,W] image to flip (torch.Tensor)'
    return torch.flip(image, [3])
    # torch.flip(input, dims) : reverse the order of a n-D tensor along given axis in dims
    
def flip_model(model, image, flip):
    '''
    flip input image and flip output inverse depth map

    Arguments:
        model: nn.Module
            module to be used
        image: torch.Tensor [B,3,H,W]
            input image
        flip: bool
            True if the flip is happening

    Returns:
        inv_depths: list of torch.Tensor [B, 1, H, W]
        list of predicted inverse depth maps
    '''
    if flip:
        return [flip_lr(inv_depth) for inv_depth in model(flip_lr(image))]
    else:
        return model(image)

'''
image gradient is defined as a directional change in image intensity.
at each pixel of the input image, a gradient measures the change in pixel intensity
in a given direction.

the gradient magnitude is used to measure how strong the change in image intensity is.
the gradient orientation is used to determine in which direction the change in intensity is pointing. 

central difference gradient. The gradient of a pixel is a weighted difference of neighboring pixels. dI/dy = (I(y+1) - I(y-1))/2

intermediate difference gradient. the gradient of a pixel is the difference between an adjacent pixel and the current pixel. dI/dy = I(y+1) - I(y)
'''
def gradient_x(image):
    '''
    calculate the gradient of an image in the x dimension

    Arguments:
        image: torch.Tensor [B,3,H,W]
            input image

    Returns:
        gradient_x: torch.Tensor [B,3,H,W-1]
            gradient of image with respect to x
    '''
    return image[:, :, :, :-1] - image[:, :, :, 1:]

def gradient_y(image):
    '''
    calculates the gradient of an image in the y dimension

    Arguments:
        image: torch.Tensor [B,3,H,W]
            input image

    Returns:
        gradient_y: torch.Tensor [B,3,H-1,W]
            gradient of image with respect to y
    '''
    returns image[:, :, :-1, :] - image[:, :, 1:, :]

def interpolate_image(image_shape, mode='bilinear', align_corners=True):
    '''
    interpolate an image to a different resolution

    Arguments:
        image: torch.Tensor [B, ?, h, w]
            image to be interpolated
        shape: tuple (H, W)
            output shape 
        mode: str
            interpolation mode
        align_corners: bool
            True if corners will be aligned after interpolation

    Returns:
        image: torch.Tensor [B, ?, H, W]
            interpolated image
    '''
    if len(shape) > 2:
        shape = shape[-2:] # (H, W)
    # if the shapes are the same, do nothing
    if same_shape(image.shape[-2:], shape):
        return image
    else:
        # interpolate image to match the shape
        return funct.interpolate(image, size=shape, mode=mode,
                                align_corners=align_corners)
        # the input dimensions are interpreted in the form: mini-batch x channels x[optional depth] x [optional height] x width
def interpolate_scales(images, shape=None, mode='bilinear', align_corners=False):
    '''
    interpolate list of images to the same shape

    Arguments:
        images: list of torch.Tensor [B, ?, ?, ?]
            images to be interpolated, with different resolutions
        shape: tuple (H,W)
            output shape
        mode: str
            interpolation mode
        align_corners: bool
            True if corners will be aligned after interpolation

    Returns:
        images: list of torch.Tensor [B,?,H,W]
            interpolated images, with the same resolution
    '''
    if shape is None:
        shape = images[0].shape
    # take last two dimensions as shape
    if len(shape) > 2:
        shape = shape[-2:] #(H,W)
        # interpolate all images 
    return [funct.interpolate(images, shape, mode=mode, align_corners=align_corners) for image in images]

### problematic function
def match_scales(image, targets, num_scales, mode='bilinear', align_corners=True):
    '''
    interpolate one image to produce a list of images with the same shape as targets

    Arguments:
        image: torch.Tensor [B, ?, h, w]
            input image
        targets: list of torch.Tensor [B,?,?,?]
            Tensors with the target resolution
        num_scales: int
            number of considered scales
        mode: str
            interpolation mode
        align_corners: bool
            True if corners will be aligned after interpolation

    Returns:
        images: list of torch.Tensor [B, ?, ?, ?]
        list of images with the same resolution as targets
    '''
    images = []
    image_shape = image.shape[-2:] #(H,W)
    for i in range(num_scales):
        target_shape = targets[i].shape
        target_shape = target_shape[-2:] #(H,W)
        # if image shape is equal to target shape
        if same_shape(image_shape, target_shape):
            images.append(image)
        else:
            images.append(interpolate_image(image, target_shape, mode=mode, align_corners=align_corners))

    return images

'''
@lru_cache(maxsize=128, typed=False)
maxsize: sets the size of the cache, the cache can store upto maxsize most recent function calls. If maxsize is set to None, the LRU feature will be disabled and the cache can grow without any limitations
typed: if set to True, function arguments of different types will be cached separately.
'''
### problematic function
### error in the original code
@lru_cache(maxsize=None)
def meshgrid(B, H, W, dtype, device, normalized=False):
    '''
    create meshgrid with a specific resolution

    Arguments:
        B: int 
            batch size
        H: int
            height
        W: int
            width
        dtype: torch.dtype
            meshgrid type
        device: torch.device
            meshgrid device
        normalized:
            True if grid is normalized between -1 and 1

    Returns:
        xs: torch.Tensor [B,H,W]
            meshgrid in dimension x
        ys: torch.Tensor [B,H,W]
            meshgrid in dimension y
    '''
    if normalized:
        # torch.linspace(start, end, steps, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False): creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive.
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype) #(W,)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype) #(H,)

    else:
        xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)

    ys, xs = torch.meshgrid([ys, xs])
    return xs.repeat([B,1,1]), ys.repeat([B,1,1])

### problematic function
### H, W has to be of same size
def image_grid(B, H, W, dtype, device, normalized=False):
    '''
    create an image grid with a specific resolution

    Arguments:
        B: int
            batch size
        H: int
            height
        W:  int
            width
        dtype: torch.dtype
            meshgrid type
        device: torch.device
            meshgrid device
        normalized: bool
            True if grid is normalized between -1 and 1

    Returns 
        grid: torch.Tensor [B, 3, H, W]
            image grid containing a meshgrid in x, y and 1
    '''
    xs, ys = meshgrid(B, H, W, dtype, device, normalized=normalized)
    ones = torch.ones_like(xs) #[B, H, W]
    # [B,H,W] [B,H,W] [B,H,W] => [B 3 H W]
    grid = torch.stack([xs, ys, ones], dim=1)
    return grid




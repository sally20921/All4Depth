# Introduction to Computer Vision and Neural Networks 
In this section, you will develop your understanding of the theory as well as learn hands-on techniques about the application of a convolutional neural network for image processing. You will learn key concepts such as image filtering, feature maps, edge detection, the convolution operation, activation functions, and the use of fully connected and softmax layers in relation to image classification and object detection. The chapter provides many hands-on examples of an end-to-end computer vision pipeline using TensorFlow, Keras, and OpenCV. The most important learning that you will take from these chapters is to develop an understanding and intuition behind different convolution operations-how images are transformed through different layers of convolutional neural networks. 

By the end of this section you will be able to do the following:
1. Understand how image filters transform an image
2. Apply various types of image filters for edge detection
3. Detect simple objects using OpenCV contour detection and Histogram of Oriented Gradients (HOG)
4. Find the similarity between objects using Scale-invariant feature transform (SIFT), Local Binary Patterns (LBP) pattern matching, and color matching.
5. Face detection using the OpenCV cascade detector
6. Input big data into a neural network from a CSV file list and parse the data to recognize columns, which cna then be fed to the neural network as *x* and *y* values.
7. Facial keypoint and facial expression recognition
8. Develop an annotation file for facial keypoints
9. Input big data into a neural network from files using the Keras data generator method
10. Construct your own neural network and optimize its parameters to improve accuracy
11. Write code to transform an image through different layers of the convolutional neural network 

# Computer Vision and TensorFlow Fundamentals
Computer vision is rapidly expanding in many different applications as traditional techniques, such as image thresholding, filtering, and edge detection, have been augmented by deep learnning methods. 

TensorFlow is a widely used, powerful machine learning tool created by Google. It has use configurable APIs available to train and build complex neural network model in your local PC or in the cloud and optimize and deploy at scale in edge devices. 

In this chapter, you will gain an understanding of advanced computer vision concepts using TensorFlow. This chapter discusses the foundational concepts of computer vision and TensorFlow to prepare you for the later, more advanced chapters of this book. We will look at how to perform image hashing and filtering. Then, we will learn about various methods of feature extraction and image retrieval. Moving on, we will learn about visual search in applications, its methods, and the challenges we might face. Then we will look at an overview of the high-level TensorFlow software and its different components and subsystems.

The topics we will be covering in this chapter are as follows:
- Detecting edges using image hashing and filtering
- Extracting features from an image
- Object detection using Contours and the HOG detector
- An overview of TensorFlow, its ecosystem, and installation

## Detecting edges using image hashing and filtering
Image hashing is a method used to find similarity between images. Hashing involves modifying an input image to a fixed size of binary vector through transformation. There are different algorithms for image hashing using different transformations: 
**Perpetual hash (phash)**: A cosine transformation
**Difference hash (dhash)**: The difference between adjacent pixels

After a hash transformation, images can be compared quickly with the Hamming distance. The Python code for applying a hash trasnformation is shown in the following code. A hamming distance of 0 shows an identical image (duplicate), whereas a larger hamming distance shows that the images are different from each other. The following snippet imports Python packages.

```
from PIL import Image
import imagehash
import distance
import scipy.spatial

hash1 = imagehash.phash(Image.open(../car1.png))
hash2 = imagehash.phash(Image.open(../car2.png))
print(hamming_distance(hash1, hash2))
```
Image filtering is a fundamental computer vision operation that modifies the input image by applying a kernel or filter to every pixel of the input image. The following are the steps involved in image filtering, starting from light entering a camera to the final transformed image:
1. Using a Bayer filter for color pattern formation
2. Creating an image vector
3. Transforming the image
4. Linear filtering-convolution with kernels
5. Mixing Gaussian and Laplacian filters
6. Detecting edges in the images

### Using a Bayer filter for color pattern formation
A Bayer filter transforms a raw image into a natural, color-processed image by applying demosaic algorithm. The image sensor consists of photodiodes, which produce electrically charged photons proportional to the brightness of the light. The photodiodes are grayscale in nature. Bayer filter are used to convert the grayscale image to color. The color image from the Bayer filter goes through an Image Signal Processing (ISP) which involves several weeks of manual adjustment of various parameters to produce desired image quality for human vision. Several research work are currently ongoing to convert the manual ISP to a CNN based processing to produce an image and then merge the CNN with image classification or object detection model to produce one coherent neural network pipeline that takes Bayer color image and detects object with bounding boxes. 

- The Bayer filter consists of Red (R), Green (G), and Blue (B) channels in a predefined pattern, such that there is twice the number of G channels compared to B and R.
- The G, R, and B channels are alternately distributed. Most channel combinations are RGGB, GRGB, RGBG. 
- Each channel will only let a specific color to pass through, the combination of colors from different channels produce a pattern as shown in the preceding image. 

### Creating an image vector
Color images are a combination of R,G, and B. Colors can be represented as an intensity value, ranging from 0 to 255. So, each image can be represented as a three-dimensional cube, with the x and y axis representing the width and height and the z axis representing three color channels (R,G,B) representing the intensity of each color. 

OpenCV is a library with built-in programming functions written for Python and C++ for image processing and object detection.

We will start by writing the following Python code to import an image, and then we will see how the image can be broken down into a NumPy array of vectors with RGB. We will then convert the image to grayscale and see how the image looks when we extract only one component of color from the image: 
```
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open('../car.jpeg')
plt.imshow(image)
image_arr = np.asarray(image)
image_arr.shape
# output: (296, 465, 4)
gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.imshow(image_arr[:, :, 0]) # red channel
plt.imshow(image_arr[:, :, 1]) # green channel
plt.imshow(image_arr[:, :, 2]) # blue channel
```
The preceding figure can be represented as a 3D volume with the following axes:
- The x axis, representing the width
- The y axis, representing the height
- Each color channel represents the depth of the image

Let's take a look at the following figure. It shows the R,G, and B pixel values for the image of the car at different x and y coordinates as a 3D volume; a higher value indicates a brighter image. 

### Transforming an image
Image transformation involves the translation, rotation, magnification, or shear of an image. If (x,y) is the coordinate of the pixel of an image, then the transformed image coordinate (u,v) of the new pixel can be represented as follows.

Image transformation is particularly helpful in computer vision for getting different images from the same image. This helps the computer develop a neural network model that is robust to translation, rotation, and shear. For example, if we only input an image of the front of a car in the convoluted neural network (CNN) during the training phase, the model will not be able to detect the image of a car rotated by 90 degrees during the test phase. 

### Linear filtering - convolution with kernels

Convolution in computer vision is a linear algebra operation of two arrays (one of them is an image and the other one is a small array) to produce a filtered image array whose shape is different than the original image array. Convolution is cumulative and associative. It can be represented mathetically as follows, and explained as follows:
- *F(x,y)* is the original image. 
- *G(x,y)* is the filtered image.
- *u* is the image kernel.

Depending on the kernel type, *u*, the output image will be different. The Python code for the conversion is as follows:
```
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open('../carshot.png')
plt.imshow(image)
image_arr = np.asarray(image)
image_arr.shape
gray = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
kernel = np.array([[-1,-1,-1],
		[2,2,2],
		[-1,-1,-1]])
blurimg = cv2.filter2D(gray, -1, kernel)
plt.imshow(blurimg, cmap='gray')
```

To the left is the input image and to the right is the image obtained by applying a horizontal kernel to the image. A horizontal kernel detects only the horizontal edges, which can be seen by the white streaks of horizontal lines. Details about the horizontal kernel can be seen in the image gradient section. 


The preceding code imports the necessary Python libraries for machine learning and computer vision work, such as NumPy to process an array, cv2 for openCV computter vision work, PIL to process images in the Python code, and Matplotlib to plot results. It then imports the image using PIL and converts it to grayscale using the OpenCV *BGR2GRAY* scale function. It creates a kernel for edge filtering using a NumPy array, blurs the image using the kernel, and then displays it using the *imshow()* function.

The filtering operation is broken down into three distinct classes:
- image smoothing
- image gradient 
- image sharpening

### Image smoothing
In image smoothing, the high-frequency noise from an image is removed by applying low-pass filter, such as the following:
- a mean filter
- a median filter
- a gaussian filter

This blurs the image and is performed by applying a pixel whose end values do not change signs and do not differ in value appreciably. 

Image filtering is typically done by sliding a box filter over an image. A box filter is represented by an n x m kernel divided by (nxm), where *n* is the number of rows and *m* is the number of columns. For a 3x3 kernel this looks as follows: 
Let's say this kernel is applied to the RGB image described previously. For reference, the 3x3 image value is shown here

#### The mean filter
The mean filter filters the image with an average value after the convolution operation of the box kernel is carried out with the image. 

#### The median filter
The median filter filters the image value with the median value after the convolution operation of the box kernel is carried out on the image. 

#### The Gaussian filter
The Gaussian kernel is represented by the following equation:
*U(i,j)* is the standard deviation of the distribution and *k* is the kernel size.

For standard deviation of *sigmoid* of 1, and 3x3 kernel (k=3), the Gaussian kernel looks as follows. In this example, when the Gaussian kernel is applied, the image is transformed as follows:

### Image filtering with OpenCV
The image filtering concepts described previously can be better understood by applying a filter to a real image. OpenCV provides a method to do that. The important code is listed inthe following snippet. After importing the image, we can add noise. Without noise, the image filtering effect can not be visualized as well. After that, we need to save the image. This is not necessary for the mean and Gaussian filter, but if we don't save the image with the median filter and import it back again, Python displays an error.

```
img = cv2.imread('car.jpeg')
imgnoise = random_noise(img, mode='s&p', amount=0.3)
plt.imsave("car2.jpg", imgnoise)
imgnew = cv2.imread('car2.jpg')
meanimg = cv2.blur(imgnew, (3,3))
medianimg = cv2.medianBlur(imgnew, 3)
gaussianimg = cv2.GaussianBlur(imgnew, (3,3), 0)
```
Note that in each of  the three cases, the filter removes the noise from the image. In this example, it appears the median filter is the most effective of the three methods in removing the noise from the image. 

## Image gradient
The image gradient calculates the change in pixel intensity in a given direction. The change in pixel intensity is obtained by performing a convolution operation on an image with a kernel. The kernel is chosen such that the two extreme rows or columns have opposite signs (positive and negative) so it produces a different operator when multiplying and summing across the image pixel. 

The image gradient described here is a fundamental concept for comptuer vision:
- The image gradient can be calculated in both the x and y directions. 
- By using the image gradient, edges and corners are determined.
- The edges and corners pack a lot of information about the shape of feature of an image.
- So the image gradient is a mechanism that converts lower-order pixel information to higher-order image features, which is used by convolution operation for image classification. 

## Image sharpening
In image sharpening, the low-frequency noise from an image is removed by applying 

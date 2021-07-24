# Reproduction of SOTA Monocular Depth Estimation Methods
```
wget -i splits/kitti_archieves_to_download.txt -P kitti_data/
cd kitti_data/
unzip "*.zip"
cd ..
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2, 1x1, 1x1 {.}.png {.}.jpg && rm {}'
```
- The above conversion command creates images with default chroma subsampling `2x2, 1x1, 1x1`. 

## Coordinate Transformations in Robotics
* In general, the location of an object in 3-D space can be specified by position and orientation values. 
* Translation nad rotation are alternative terms for position and orientation. 

### Axis-Angle
* A rotation in 3-D space by a scalar rotation and around a fixed axis defined by a vector
* 1-by-3 unit vector and a scalar angle combined as 1-by-4 vector
```
axang = [0 1 0 pi/2]
```

### Euler Angles
* Euler angles are three angles that describe the the orientation of a rigid body. 
* Each angle is a scalar rotation around a given coordinate frame axis. 
* The 'ZYZ' axis order is commonly used for robotics applications. 
* Knowing which axis order you use is important for applying the rotation to points and in converting to other representations.
* 1-by-3 vector of scalar angles
* A rotation around the y-axis of pi would be expressed as:
```
eul = [0 pi 0]
```

### Homogeneous Transformation Matrix
* A homogeneous transformation matrix combines a translation and rotation into one matrix.
* 4-by-4 matrix
* a rotation of a angle a around the y-axis and a translation of 4 units along the y-axis would be expressed as:
```
# tform = [[cosa 0 sina 0]
	   [0 1 0 4]
	   [-sina 0 cosa 0]
	   [0 0 0 1]]
```
* You should pre-multiply your transformation matrix with your homogeneous coordinates, which are represented as a matrix of row vectors (n-by-4 matrix of points). 

### Quarternion
* A quaternion is a four-element vector with a scalar rotation and 3-element vector. 
* Quaternions are advantageous because they avoid singularity issues that are inherent in other representations. 
* The first element is a scalar to normalize the vector with the three other values, *[x,y,z]* defining the axis of rotation. 
* 1-by-4 vector
* a rotation of pi/2 around the y-axis would be represented as `quat = [0.7071 0 0.7071 0]`.

### Rotation Matrix
* A rotation matrix describes a rotation in 3-D space. 
* It is a square, orthonormal matrix with a determinant of 1.
* 3-by-3 matrix 
* a rotation of a degrees around x-axis would be 
```
# rotm = [[1 0 0]
	 [0 cos a -sin a]
	 0 sin a cos a]]
```
* You should pre-multiply your rotation matrix with your coordinates, which are represented as a matrix of row vectors (n-by-3 matrix of points). 

### Translation Vetor
* A translation vector is represented in 3-D Euclidean space as Cartesian coordinates. 
* It only involves coordinate translation applied equally to all points. 
* There is no rotation involved. 
* 1-by-3 vector
* a translation by 3 units along the x-axis and 2.5 units along the z-axis would be expressed as`trvec = [3 0 2.5]`.


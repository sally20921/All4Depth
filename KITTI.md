- Notice that we are using homogeneous coordinates, in that we have added a 1 to the end of a vector in 3D space. This makes the application of the rotational and translational components of the transformation convenient in one multiplication step. This trick will continue to come in handy later.

- Now that we have our array of ground truth poses, we can look at true trajectory of the car through the sequence. We don't need to be concerned with the roations here, since their effects are implicit in the position of the camera origin over time, which is represented by the fourth column of each transformation matrix. 

- On the subject of projection matrices, let's take a moment to discuss what information is contained in these 3x4 matrices, and how it is useful to us. These matrices contain intrinsic information about the camera's focal length and optical center, which we will discuss later. 

- Further, they also contain transformation information which relates each camera's coordinate frame to the global coordinate frame.

- A projection matrix takes 3D coordinates in the global coordinate frame and projects them onto the image plane of the camera. 

- Let us first break down the relationship between a camera's projective matrix and its intrinsic and extrinsic matrices.

- A projection matrix *P* is the dot product of the intrinsic and extrinsic matrices of a camera. The intrinsic matrix *K* contains the focal length and optical center parameters, and the extrinsic matrix *R|t* contains the pose of the camera in the same form we saw earlier with the ground truth poses of the car: a 3x3 rotation matrix horizontally stacked with a 3x1 translation vector. 

- Breaking down the *P* matrix into intrinsic and extrinsic camera matrices in the above equation then provides us with the following, which is a more explicit description of the process of projecting a 3D point in any coordinate frame into the pixel coordinate frame of the camera. 

- Now, remember that the projection matrices from the calibration file are the camera projection matrices for each camera AFTER RECTIFICATION in terms of the stereo rig. Normally, a camera's projection matrix take a 3D point in a global coordinate frame and projects it onto pixel coordinate on THAT camera's image frame. Rectified projection matrices are the opposite, and are designed to map points each camera's coordinate frame onto one single image plane: that of the left camera. This means they are going in the opposite direction, as these matrices are taking 3D points from the coordinate frame of the camera they are associated with, and projecting them onto the image plane of the left camera. If phrased in terms of normal projection matrix logic, *P0* through *P1* are basically 4 different projection matrices for the same (left grayscale) camera, considering 4 different global coordinate frames for each ofthe cameras. 

- Each rectified projection matrix will take *(X,Y,Z,1)* homogeneous coordinates of 3D points in the associated sensor's coordinate frame and translate them to pixel lcoations *(u,v,1)* on the image plane of the left grayscale camera. 

- As an exercise, let's turn this rectified projection matrix into a regular projection matrix for the right grayscale camera. We do this by decomposing the matrix, making the extrinsic matrix homogeneous, inverting it, then recombining the intrinsics and inverted extrinsic. 

- Now that we know what these calibration matrices are actually telling us, and that they are all in reference to the left grayscale camera's image plane, let's go back and look again at the equation which projects 3D points from a given coordinate frame onto the image plane of a camera, and talk more about the *lambda* value.



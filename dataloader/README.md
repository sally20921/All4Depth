# The KITTI vision benchmark suite: Raw Data Recordings
- the camera images are stored in the following directories:
* *img_00*: left rectified grayscale image sequence
* *img_01*: right rectified grayscale image sequence
* *img_02*: left rectified color image sequence
* *img_03*: right rectified color image sequence

## Velodyne 3D laser scan data
- The velodyne point clouds are stored in the folder *velodyne_points*. To save space, all scans have been stored as Nx4 float matrix into a binary file using the following code:
```
stream = fopen(dst_file.str(), "wb");
fwrite(data, sizeof(float), 4*num, stream);
fclose(stream)
```

- Here, data contains `4*num` values, where the first 3 values correspond to  x, y, and z, and the last value is the reflectance information.
- All scans are stored row-aligned, meaning that the first 4 values correspond to the first measurement. 

## example transformation
- As the transformations sometimes confuse people, here we give short example of how points in the velodyne system can be transformed into the camera left coordinate system.

- In order to transform a homogeneous point *X = [x y z 1]* from the velodyne coordinate system to a homogeneous point *Y = [u v 1]* on image plane of camera xx, the following transformation has to be applied:

* *Y = P_rect_xx * R_rect_00 * (R|T)_velo_to_cam * X*
- To transform a point *X* from GPS/IMU coordinate to the image plane:
* *Y = P_rect_xx * R_rect_00 * (R|t)_velo_to_cam * (R|T)_imu_to_velo * X*

- The matrices are:
- *P_rect_xx* (3x4): rectified cam 0 coordinates -> image plane
- *R_rect_00* (4x4): cam 0 coordinates -> rectified cam 0 coord
- *(R|T)_velo_to_cam* (4x4): velodyne coordinates -> cam 0 coordinates
- *(R|T)_imu_to_velo* (4x4): imu coordinates -> velodyne coordinates

- Note that the (4x4) matrices above are padded with zeros and:
*R_rect_00(4,4) = (R|T)_velo_to_cam(4,4) = (R|T)_imu_to_velo(4,4) = 1*.


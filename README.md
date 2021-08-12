# *MonoDepth to ManyDepth*: Self-Supervised Depth Estimation on Monocular Sequences

![merge-athens](https://user-images.githubusercontent.com/38284936/128589385-cd2e68a5-8f27-4aaa-abfe-85cbf3fffe21.png)
![Trevi-merge](https://user-images.githubusercontent.com/38284936/128589386-12374c8c-bdd8-4545-8b10-1acfdec9039a.png)

1. Dataset 
	- Dense Depth for Autonomous Driving (DDAD)
	- KITTI Eigen Split
	```
	wget -i splits/kitti_archieves_to_download.txt -P kitti_data/
	cd kitti_data/
	unzip "*.zip"
	cd ..
	find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2, 1x1, 1x1 {.}.png {.}.jpg && rm {}'
	```
	- The above conversion command creates images with default chroma subsampling `2x2, 1x1, 1x1`. 

2. Problem Setting

	while specialist hardware can give per-pixel depth, a more attractive approach is to only require a single RGB camera.
	
	train a deep network to map from an input image to a depth map 
	
	![image](https://user-images.githubusercontent.com/38284936/128327929-5b9b0d88-8d8a-4832-b900-e53e21ea0edf.png)
	
	![image](https://user-images.githubusercontent.com/38284936/128327970-7cfa78e0-dba0-4619-b200-b9ad3306e4eb.png)

3. Methods
	- Geometry Models
	  
	  The simplest representation of a camera an image plane at a given position and orientation in space. 
	  
	  ![image](https://user-images.githubusercontent.com/38284936/128328232-1f5072ba-5e24-4415-b6ef-852de9fb17c0.png)
	  
	  The pinhole camera geometry models the camera with two sub-parameterizations, intrinsic and extrinsic paramters. Intrinsic parameters model the optic component (without distortion), and extrinsic model the camera position and orientation in space. This projection of the camera is described as:
	  
	  ![image](https://user-images.githubusercontent.com/38284936/128328314-bab98131-be0b-4a3a-82c2-48130e63b4cd.png)
	  
	  A 3D point is projected in a image with the following formula (homogeneous coordinates):
	  
	  ![image](https://user-images.githubusercontent.com/38284936/128328372-313475dd-bad3-4cd7-8775-fee689b733a0.png)
	  
	- Cross-View Reconstruction
	
	frames the learning problem as one of novel view-synthesis, by training a network to predict the appearance of a target image from the viewpoint another image using depth (disparity)
	
	formulate the problem as the minimization of a photometric reprojection  error at training time
	
	![image](https://user-images.githubusercontent.com/38284936/128328569-28c573e5-7a63-4fa2-93e4-9da028ceabad.png)
	
	![image](https://user-images.githubusercontent.com/38284936/128328609-78f198b8-1bf9-4599-9410-769143e46f52.png)
	
	![image](https://user-images.githubusercontent.com/38284936/128328628-54f2c6f1-cb40-4fd3-866d-4e36f699b1f7.png)
	
	Here. pe is a photometric reconstruction error, proj() are the resulting 2D coordinates of the projected depths Dₜ in the source view and <> is the sampling operator.  For simplicity of notation we assume the pre-comuted intrinsics K of all views are identical, though they can be different. α is set to 0.85.
	
	![image](https://user-images.githubusercontent.com/38284936/128328686-a7a592f1-223f-4183-8ec5-b8c59c03dbf3.png)
	
	consider the scene structure and camera motion at the same time, where camera pose estimation has a positive impact on monocular depth estimation. these two sub-networks are trained jointly, and the entire model is constrained by image reconstruction loss similar to stereo matching methods. 
	formulate the problem as the minimization of a photometric reprojection  error at training time
	formulate the problem as the minimization of a photometric reprojection  error at training time

3. Folder
```
dataset/
    2011_09_26/
    ...
    ...
model_dataloader/
model_layer/
model_loss/
model_save/
model_test.py
model_train.py
model_parser.py
model_utility.py
```

4. Packages
```
apt-get update -y
apt-get install moreutils
or
apt-get install -y moreutils
```

5. Training
```
python model_train.py --pose_type separate --datatype kitti_eigen_zhou
python model_train.py --pose_type separate --datatype kitti_benchmark
```

6. Test
```
python model_test.py
```

7. evaluation
```
kitti_eigen_zhou 
abs_rel   sqrt_rel  rmse      rmse_log  a1        a2        a3
0.125     0.977     4.992     0.202     0.861     0.955     0.980

kitti_eigen_benchmark
abs_rel   sqrt_rel  rmse      rmse_log  a1        a2        a3
0.104     0.809     4.502     0.182     0.900     0.963     0.981
```

## Padding
### What is padding and why do we need it?

![Screen Shot 2021-08-12 at 8 33 57 AM](https://user-images.githubusercontent.com/38284936/129116773-479b1e81-211c-4982-974d-944a95853dca.png)

* What is a feature map? that's the yellow block in the image. 
* It's a collection of *N* one-dimensional "maps" that each represent a particular "feature" that the model has spotted within the image. 
* why convolutional layers are known as feature extractors

* How 






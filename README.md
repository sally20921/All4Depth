## All4Depth: A PyTorch Collection of Self-Supervised Depth Estimation Methods 

### Installation
- You need a machine with recent NVIDIA drivers and GPU with at least 6GB of memory (more for the bigger models at higher resolution).
- Use the docker to have a reproducible environment.

- This is how you can set up your environment:
 * If you don't have nvidia-docker, go to the `Makefile` and change it to `docker --gpus=all`. You can optionally add `-v /SSD/seri:/data/datasets` to mount your dataset.
```
git clone https://github.com/sally20921/All4Depth
cd All4Depth
make docker-build
```

### Datasets
- Datasets are assumed to be downloaded in `/data/datasets/$dataset-name$`.
- Symbolic links are also allowed.
- For example, `/data/datasets/KITTI_raw` or `/data/datasets/KITTI_tiny`

1. KITTI tiny dataset for testing 
- For simple tests, tiny version of KITTI is provided.
```
curl -s https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/datasets/KITTI_tiny.tar | tar -xv -C /data/datasets/
```

2. KITTI raw dataset for training
- For convenience, the standard splits used for training and evaluation are provided.
- Go to the folder `/data/datasets/KITTI_raw/data_splits`
- The pre-computed ground-truth maps are also provided for supervised training. 

```
# the kitti raw dataset with standard splits 
curl -s https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/datasets/KITTI_raw.tar | tar -xv -C /data/datasets/

# the original ground truth maps
https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/depth_maps/KITTI_raw_velodyne.tar.gz | tar -xv -C /data/datasets/

# the improved ground truth maps
https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/depth_maps/KITTI_raw_groundtruth.tar.gz | tar -xv -C /data/datasets/
```

### Running Commands Inside the Container
- You can start the container in interative mode with the following lines:
```
make docker-start-interative
```

#### Evaluation

- To evaluate a trained model, you need to provide a `.ckpt` checkpoint, followed by a `.yaml` config file that overrides the configuration stored in the checkpoint.

- The pretrained model can be downloaded via this commandline:
```
# first create a checkpoint folder in your project
mkdir /workspace/All4Depth/ckpt

# download self-supervised, 192x640, trained on raw KITTI model
curl -o /workspace/All4Depth/ckpt https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/PackNet01_MR_selfsup_K.ckpt 

# download self-supervised, trained on Cityscapes
curl -o /workspace/All4Depth/ckpt https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/models/PackNet01_MR_selfsup_CS.ckpt
```
- The pretrained model followed the Eigen et al. protocol for training and evaluation, with Zhou et al.'s preprocessing to remove static training frames.

- Then create a `.yaml` config file in your project configs folder via `touch /workspace/All4Depth/configs/user.yaml` like the example below:
```yaml
checkpoint:
        filepath: /workspace/All4Depth/ckpt
model: 
        checkpoint_path: /workspace/All4Depth/ckpt/PackNet01_MR_selfsup_K.ckpt
        depth_net: 
                checkpoint_path: /workspace/All4Depth/depth_net.ckpt
        pose_net:
                checkpoint_path: /workspace/All4Depth/pose_net.ckpt
```
- Note that providing `depth_net` and `pose_net` are optional.
- Defining a checkpoint to the model itself will include all sub-networks.
- Also keep in mind that setting the model checkpoint will overwrite depth and pose checkpoints.


- For reference, `.yaml` files are provided in `/workspace/All4Depth/configs` to train and test the model.

- The following command overrides the configuration stored in the checkpoint.
```
python3 scripts/eval.py --checkpoint ckpt/PackNet01_MR_selfsup_K.ckpt --config configs/user.yaml
```

- To directly run inference on a single image or folder, do:
```
python3 scripts/infer.py --checkpoint ckpt/PackNet01_MR_selfsup_K.ckpt --input assets/kitti.png --output assets/
```
- You can optionally provide the image shape with the flag `--image_shape`.

#### Training

- Training can be done by passing your `.yaml` config file.
```
python3 scripts/train.py configs/user.yaml
```
- If you pass a config file without providing the checkpoint path, training will start from scratch.
- If you specify a `.ckpt` path, training will continue from the current checkpoint state. 
- Every aspect of the training configuration can be controlled by modifying the yaml config file. 
- This includes the model configuration as well as the depth and pose networks configuration, optimizers and schedulers and datasets.

#### WANDB logging and AWS checkpoint syncing

- To enable WANDB logging and AWS checkpoint syncing, set the corresponding configuration parameters in `configs/user.yaml`.

```yaml
wandb:
        dry run: True # WANDB dry run (not logging)
        name: ''      # WANDB run name
        project: os.environment.get("WANDB_PROJECT", "") # WANDB project
        entity: os.environment.get("WANDB_ENTITY", "")   # WANDB entity
        tags: [] 
        dir: '' # WANDB save folder
checkpoint:
        s3_path: ''     # s3 path for AWS model syncing
        s3_frequency: 1 # How often to s3 sync
```

### Additional Tips & Datasets
- If you encounter out of memory issues, try a lower `batch_size` parameter in the config file. 
- You can also train/test the model using DDAD (Dense Depth for Autonomous Driving) or OmniCam. Download them via:
```
# DDAD_tiny
curl -s https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/datasets/DDAD_tiny.tar | tar -xv -C /data/datasets/

# DDAD
curl -s https://tri-ml-public.s3.amazonaws.com/github/DDAD/datasets/DDAD.tar | tar -xv -C /data/datasets/

# omnicam
curl -s https://tri-ml-public.s3.amazonaws.com/github/packnet-sfm/datasets/OmniCam.tar | tar -xv -C /data/datasets/
```
### Results

|        Model        | Abs.Rel. | Sqr.Rel. | RMSE  | RMSElog |
|:-------------------:|:--------:|----------|-------|---------|
| SSL, 192x640, KITTI |   0.111  |   0.800  | 4.576 | 0.189   |


### Reference
```
@inproceedings{packnet,
  author = {Vitor Guizilini and Rares Ambrus and Sudeep Pillai and Allan Raventos and Adrien Gaidon},
  title = {3D Packing for Self-Supervised Monocular Depth Estimation},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  primaryClass = {cs.CV}
  year = {2020},
}

@inproceedings{packnet-san,
  author = {Vitor Guizilini and Rares Ambrus and Wolfram Burgard and Adrien Gaidon},
  title = {Sparse Auxiliary Networks for Unified Monocular Depth Prediction and Completion},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  primaryClass = {cs.CV}
  year = {2021},
}

@inproceedings{shu2020featdepth,
  title={Feature-metric Loss for Self-supervised Learning of Depth and Egomotion},
  author={Shu, Chang and Yu, Kun and Duan, Zhixiang and Yang, Kuiyuan},
  booktitle={ECCV},
  year={2020}
}

@article{monodepth2,
  title     = {Digging into Self-Supervised Monocular Depth Prediction},
  author    = {Cl{\'{e}}ment Godard and
               Oisin {Mac Aodha} and
               Michael Firman and
               Gabriel J. Brostow},
  booktitle = {The International Conference on Computer Vision (ICCV)},
  month = {October},
year = {2019}
}
``


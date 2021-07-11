from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__) # the directory that options.py resides in

class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="monodepth options")
        self.parser.add_argument("--data_path", type=str, help="path to the training data", default=os.path.join(file_dir, "kitti_data"))
        self.parser.add_argument("--log_dir", type=str, help="log directory", default=os.path.join(file_dir, "log"))

        self.parser.add_argument("--model_name", type=str, help="the name of the folder to save the model in", default=os.path.join(file_dir, "ckpt"))
        self.parser.add_argument("--split", type=str, help="which training split to use", choices=["eigen_zhou", "eigen_full", "odom", "benchmark"],
                default="eigen_zhou")
        self.parser.add_argument("--num_layers", type=int, help="number of resnet layers", choices=[18, 34, 50, 101, 152], default=18)
        self.parser.add_argument("--dataset", type=str, help="dataset to train on", default="kitti", choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        self.parser.add_argument("--png", help="if set, train from raw KITTI png files (instead of jpgs)", action="store_true")
        self.parser.add_argument("--height", type=int, help="input image height", default=192)
        self.parser.add_argument("--width", type=int, help="input image width", default=640)
        self.parser.add_argument("--disparity_smoothness", type=float, help="disparity smoothness weight", default=1e-3)
        self.parser.add_argument("--scales", nargs="+", type=int, help="scale used in the loss", default=[0,1,2,3])
        self.parser.add_argument("--min_depth", type=float, help="minimum dpeth", default=0.1)
        self.parser.add_argument("--max_depth", type=float, help="maximum depth", default=100.0)





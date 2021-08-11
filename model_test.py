import os
import sys
import argparse
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from model_parser import *
from model_utility import *
from model_dataloader import *
from model_layer import *
from model_loss import *




device   = 'cuda:0' if torch.cuda.is_available() else 'cpu'
testpath = {
    "kitti_benchmark": "./splits/kitti_test/kitti_benchmark_test_files.txt",
    "kitti_eigen_benchmark": "./splits/kitti_test/kitti_eigen_benchmark_test_files.txt",
    "kitti_eigen_test": "./splits/kitti_test/kitti_eigen_test_files.txt"}


def load_weights(args):
    encoder_weights = torch.load(args.encoder_path)
    decoder_weights = torch.load(args.decoder_path)

    model = {}
    model["encoder"] = ResnetEncoder(18, False)
    model["decoder"] = DepthDecoder(model["encoder"].num_ch_enc)

    model["encoder"].load_state_dict(
        {k: v for k, v in encoder_weights.items() if k in model["encoder"].state_dict()})
    model["decoder"].load_state_dict(decoder_weights)

    for key in model:
        model[key].to(device)
        model[key].eval()
    return model


def inference(args):
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80.0
    filename = readlines(testpath["kitti_{}_test".format(args.splits)])
    dataset  = KITTIMonoDataset("./dataset/kitti", filename, False, [0], ".jpg", 1)
    loader   = DataLoader(dataset, batch_size = 16, shuffle = False, drop_last = False)
    print(">>>   Testset length {}, Batch iteration {}".format(len(filename), loader.__len__()))

    model = load_weights(args)
    print(">>>   Loaded model")

    # loader를 iter해서 이미지로 디스패리티를 출력하고
    # GT 뎁스와 함께 넘파이 리스트로 변환
    predction_list    = []
    grount_truth_list = []
    return_result     = []
    with torch.no_grad():
        for data in tqdm(loader):
            color_image  = data[("color", 0, 0)].to(device)
            ground_truth = data[("depth", 0)].cpu()[:, 0].numpy().astype(np.float32)
            
            outputs      = model["decoder"](model["encoder"](color_image))
            pred_disp, _ = disparity2depth(outputs[("disp", 0)], MIN_DEPTH, MAX_DEPTH)
            pred_disp    = pred_disp.cpu()[:, 0].numpy()
            
            return_result.append(outputs)
            predction_list.append(pred_disp)
            grount_truth_list.append(ground_truth)
    predction_list    = np.concatenate(predction_list)
    grount_truth_list = np.concatenate(grount_truth_list)

    errors_list = []
    for index in tqdm(range(len(predction_list))):
        pred_disparity = predction_list[index]
        ground_truth   = grount_truth_list[index]
        height, width  = ground_truth.shape

        pred_disparity = cv2.resize(pred_disparity, (width, height))
        pred_depth     = 1 / pred_disparity

        if args.splits == "eigen":
            mask = np.logical_and(ground_truth > MIN_DEPTH, ground_truth < MAX_DEPTH)
            crop = np.array([153, 371, 44, 1197]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)
        else:
            mask = ground_truth > 0.
        
        pred_depth   = pred_depth[mask]
        pred_depth   *= 1
        ground_truth = ground_truth[mask]

        if args.median == True:
            pred_depth *= np.median(ground_truth) / np.median(pred_depth)

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
        errors_list.append(compute_depth_error(ground_truth, pred_depth, "numpy"))
    mean_errors = np.array(errors_list).mean(0)

    print(">>>   abs_rel   sqrt_rel  rmse      rmse_log  a1        a2        a3")
    print(">>>" + ("   {:4.3f}  " * 7).format(*mean_errors.tolist()))
    return return_result




if __name__ == "__main__":
    weight = {
        "monodepth2 192x640 with pt": {
            "encoder": "./model_save/monodepth2/mono_640x192/encoder.pth",
            "decoder": "./model_save/monodepth2/mono_640x192/depth.pth",},
        "monodepth2 192x640 w/o pt": {
            "encoder": "./model_save/monodepth2/mono_no_pt_640x192/encoder.pth",
            "decoder": "./model_save/monodepth2/mono_no_pt_640x192/depth.pth"},
        "separate_benchmark 192x640": {
            "encoder": "./model_save/separate_benchmark/encoder20.pt",
            "decoder": "./model_save/separate_benchmark/decoder20.pt"},
        "separate_eigen_zhou 192x640": {
            "encoder": "./model_save/separate_eigen_zhou/encoder20.pt",
            "decoder": "./model_save/separate_eigen_zhou/decoder20.pt"}}
    def options():
        parser = argparse.ArgumentParser(description = "Input optional guidance for training")
        parser.add_argument("--datapath",
            default = "./dataset/kitti",
            type = str,
            help = "훈련 폴더가 있는 곳")
        parser.add_argument("--splits",
            default = "eigen",
            type = str,
            help = ["eigen", "eigen_benchmark"])
        parser.add_argument("--encoder_path",
            default = weight["separate_eigen_zhou 192x640"]["encoder"],
            type = str,
            help = "Encoder weight path")
        parser.add_argument("--decoder_path",
            default = weight["separate_eigen_zhou 192x640"]["decoder"],
            type = str,
            help = "Decoder weight path")
        parser.add_argument("--median",
            default = True,
            type = str,
            help = "median scaling option")
        args = parser.parse_args()
        return args
    return_result = inference(options())

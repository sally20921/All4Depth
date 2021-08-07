from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import numpy as np
from tqdm import tqdm

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



# Templet
templet1 = "Epoch {0:1d}   Batch {1:1d}     LR {2:0.4f}   Scheduler {3:1d}   Smoothness {4:1.3f}"
templet2 = "Input ({0:1d}, {1:1d})        min-max Depth ({2:0.1f} {3:0.1f})"
templet3 = "Epoch train iter {0:1d}   Epoch valid iter {1:1d}   Total train iter {2:1d}   Total valid iter {3:1d}"


# pytorch randomnetss
def pytorch_randomness(random_seed = 777):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU



class trainer(object):
    def __init__(self, options):
        pytorch_randomness()
        self.device           = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(">>>   Using of CUDA  :  ", self.device)
        self.opt              = options
        self.epoch            = options.epoch
        self.batch_size       = options.batch
        self.learning_rate    = options.learning_rate
        self.scheduler_step   = options.scheduler_step
        self.min_depth        = options.min_depth
        self.max_depth        = options.max_depth

        # 아래로는 이미지 사이즈와 뎁스 옵션에 대한 정의 (for KITTI)
        # self.frame_idx        = options.frame_idx | 이 부분은 기존 코드 
        self.frame_ids        = options.frame_ids    # [-2, -1, 0, 1, 2]
        self.num_frames       = len(self.frame_ids)  # len([-2, -1, 0, 1, 2])
        if options.pose_frames == "all":
            self.num_pose_frames = self.num_frames
        else:
            self.num_pose_frames = 2

        self.dataset = self.opt.datapath.split("/")[2]
        print(">>>   Using dataset  :  ", self.dataset)
        if self.dataset == "kitti":
            self.original_scale   = (375, 1242)
            self.default_scale    = [(320, 1024), (192, 640), (96, 320), (48, 160)]
        elif self.dataset == "cityscapes":
            self.original_scale   = (1024, 2048)
            self.default_scale    = [(512, 1024), (256, 512), (128, 256), (64, 128)]

        self.scale            = options.scale        # 0, 1, 2, 3 중 하나, default_scale에서 어떤 스케일을 쓸지 선택
        self.scales           = options.scales       # 스케일의 범위 [0, 1, 2, 3] (현재 지정한 self.scale로부터 시작)
        self.num_scales       = len(options.scales)  # [0, 1, 2, 3]의 길이는 4
        self.resolution       = [(self.default_scale[self.scale][0]//(2**i), 
                                  self.default_scale[self.scale][1]//(2**i))
                                  for i in self.scales]

        self.height           = self.resolution[0][0] # 학습에 사용할 4개의 스케일 중 가장 앞 스케일을 모델에 입력
        self.width            = self.resolution[0][1]
        self.num_layers       = options.num_layers    # 레즈넷 버전 18 or 36 or 50, 디폴트 18
        self.weight_init      = options.weight_init   # "True or False"
        self.pose_type        = options.pose_type     # 포즈 네트워크, "posecnn", "separate_resnet", shared_resnet"

        # 1. train, valid 데이터 파일 경로 확보
        self.datapath  = self.opt.datapath
        self.filepath  = self.opt.splits + "/" + self.opt.datatype + "/{}_files.txt"
        train_filename = readlines(self.filepath.format("train"))
        valid_filename = readlines(self.filepath.format("val"))
        # train_filename = train_filename[:300]
        # valid_filename = valid_filename[:60]

        self.train_dataloader = self.definition_dataloader(
            train_filename, is_training = True, shuffle = True, mode = "train")
        self.valid_dataloader = self.definition_dataloader(
            valid_filename, is_training = False, shuffle = False, mode = "val")

        # train_dataset = KITTIRAWDataset(
        #     self.datapath, train_filename, self.height, self.width, self.frame_ids, 4, is_train = True, img_ext = ".jpg")
        # self.train_dataloader = DataLoader(
        #     train_dataset, self.batch_size, True, num_workers = self.opt.num_workers, pin_memory = True, drop_last = True)
        # valid_dataset = KITTIRAWDataset(
        #     self.datapath, valid_filename, self.height, self.width, self.frame_ids, 4, is_train = False, img_ext = ".jpg")
        # self.valid_dataloader = DataLoader(
        #     valid_dataset, self.batch_size, True, num_workers = self.opt.num_workers, pin_memory = True, drop_last = True)
        
        self.train_length, self.valid_length = len(self.train_dataloader), len(self.valid_dataloader)
        self.train_iteration = self.train_length * self.epoch
        self.valid_iteration = self.valid_length * self.epoch
        print(">>> Setting dataloader")

        # 2. 모델과 모델 파라미터, BackPropagation과 Project3D를 로드
        self.model      = {}
        self.parameters = []
        self.definition_model()
        self.backward_project = Depth2PointCloud(self.batch_size, self.height, self.width)
        self.forward_project  = PointCloud2Pixel(self.batch_size, self.height, self.width)
        self.backward_project.to(self.device)
        self.forward_project.to(self.device)
        print(">>> Setting model")

        # 3. 모델 로스 (SSIM) 옵티마이저, 스케쥴러 로r드
        self.reprojection_loss  = ReprojectionLoss()
        self.smooth_loss        = SmoothLoss()
        self.reprojection_loss.to(self.device)
        self.smooth_loss.to(self.device)

        self.model_optimizer   = torch.optim.Adam(self.parameters, self.learning_rate)
        self.model_scheduler   = torch.optim.lr_scheduler.StepLR(self.model_optimizer, self.scheduler_step)
        self.depth_losess_name = ["loss", "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]
        print(">>> Setting Loss function & Optimizer")
        print(" ")
        
        print("Pose Network    ", self.opt.pose_type)
        print("Dataset type    ", self.opt.datatype)
        print(templet1.format(
            self.epoch, self.batch_size, self.learning_rate, self.scheduler_step, self.opt.disparity_smoothness))
        print(templet2.format(self.height, self.width, self.min_depth, self.max_depth))
        print(templet3.format(
            self.train_length, self.valid_length, self.train_iteration, self.valid_iteration))
        print(" ")
        print(">>> >>> >>> Training Start")

    

    
    def definition_dataloader(self, filename, is_training, shuffle, mode): # 키티 데이터를 불러오는 함수
        if self.dataset == "kitti":
            dataset = KITTIMonoDataset(
                self.datapath, filename, is_training, self.frame_ids, ext = ".jpg",  scale = self.opt.scale)
        elif self.dataset == "cityscapes":
            dataset = CityscapesMonoDataset(
                self.datapath, filename, is_training, self.frame_ids, mode = mode, ext = ".jpg", scale = self.opt.scale)

        dataloader = DataLoader(dataset, self.batch_size, shuffle, num_workers = self.opt.num_workers, drop_last = True)
        return dataloader


    def definition_model(self):
        # 뎁스 인코더, 디코더, 포즈 네트워크, 마스크 네트워크 로드하는 함수
        self.model["encoder"] = ResnetEncoder(num_layers = self.num_layers, pretrained = self.weight_init)
        self.model["decoder"] = DepthDecoder(num_ch_enc = self.model["encoder"].num_ch_enc, scales = self.scales)
        
        if self.pose_type == "posecnn":
            self.model["pose_decoder"] = PoseCNN(num_input_frames = self.num_pose_frames)

        elif self.pose_type == "shared":
            self.model["pose_decoder"] = PoseDecoder(self.model["encoder"].num_ch_enc, self.num_pose_frames)

        elif self.pose_type == "separate":
            self.model["pose_encoder"] = ResnetEncoder(self.num_layers, self.weight_init, self.num_pose_frames)
            self.model["pose_decoder"] = PoseDecoder(self.model["pose_encoder"].num_ch_enc, num_input_features = 1, num_frames_to_predict_for = 2)

        for key in self.model:
            self.model[key] = self.model[key].to(self.device)
            self.parameters += list(self.model[key].parameters())

    
    
    
    def model_train(self): # 훈련 루프
        # 에포크마다 평균 로스와 평균 메트릭을 보관할 딕셔너리 -> 이후 저장할 것
        epoch_train = {key: [] for key in self.depth_losess_name}
        epoch_valid = {key: [] for key in self.depth_losess_name}

        for epoch in range(self.epoch):
            # 배치 데이터마다 로스와 메트릭을 보관할 딕셔너리, 배치 마다 concat을 해서 평균 후 출력
            batch_train = {key: [] for key in self.depth_losess_name}
            batch_valid = {key: [] for key in self.depth_losess_name}

            self.set_train()
            for _, train_inputs in tqdm(enumerate(self.train_dataloader)):
                train_outputs, train_loss = self.batch_process(train_inputs)

                self.model_optimizer.zero_grad()
                train_loss["loss"].backward()
                self.model_optimizer.step()
                # 배치 데이터 로스 누적
                batch_train = self.compute_metric(train_inputs, train_outputs, train_loss, batch_train)
            
            self.set_valid()
            for _, valid_inputs in tqdm(enumerate(self.valid_dataloader)):
                with torch.no_grad():
                    valid_outputs, valid_loss = self.batch_process(valid_inputs)
                    # 배치 데이터 로스 누적
                    batch_valid = self.compute_metric(valid_inputs, valid_outputs, valid_loss, batch_valid)
            

            self.model_scheduler.step()
            # 한 에포크를 다 돌았다면, 누적한 배치 데이터의 로스를 평균하여 에포크 딕셔너리에 저장
            for key in self.depth_losess_name:
                epoch_train[key].append(np.mean(batch_train[key]))
                epoch_valid[key].append(np.mean(batch_valid[key]))

            self.model_print(epoch, batch_train, batch_valid) # 보관하고 있는 미니 배치 수를 로그내서 출력할 예정
            self.model_save(epoch, epoch_train, epoch_valid)
        print("... ... ... Training End")


    def model_print(self, epoch, train_log, valid_log):
        print("EPOCH   {0}".format(epoch+1))
        print("Train Log", end = " ")
        for key in self.depth_losess_name:
            print("  {} {:0.3f}".format(key, np.mean(train_log[key])), end = " ")
        print("\n")
        print("Valid Log", end = " " )
        for key in self.depth_losess_name:
            print("  {} {:0.3f}".format(key, np.mean(valid_log[key])), end = " ")
        print("\n")


    def model_save(self, epoch, train_log, valid_log):
        save_directory = os.path.join("./model_save", self.opt.save_name)
        if not os.path.isdir(save_directory):
            os.makedirs(save_directory)
        
        if (epoch+1) % 10 == 0: # epoch가 특정 조건을 만족시키는 조건문, 뎁스 인코더, 디코더 모델 저장
            for key in self.model:
                torch.save(self.model[key].state_dict(),
                    os.path.join(save_directory, key + str(epoch+1) + ".pt"))
        
        if (epoch+1) == self.epoch:
            for key in self.model:
                torch.save(self.model[key].state_dict(),
                    os.path.join(save_directory, key + str(epoch+1) + ".pt"))

            for key in self.depth_losess_name: # 모델의 로그 기록 저장
                np.save(os.path.join(save_directory, key + str(epoch+1) + ".npy"), train_log[key])
                np.save(os.path.join(save_directory, key + str(epoch+1) + ".npy"), valid_log[key])


    
    
    def batch_process(self, inputs): # 배치 데이터마다 처리
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        
        if self.pose_type == "posecnn": # posecnn 타입의 경우는 이미지 한 장을 뎁스 네트워크에 포워드
            features = self.model["encoder"](inputs[("color_aug", 0, 0)])
            outputs  = self.model["decoder"](features)
        
        elif self.pose_type == "shared": # shared 이면 모든 프레임을 concat해서 포워드
            all_frames   = torch.cat([inputs[("color_aug", frame_id, 0)] for frame_id in self.frame_ids])
            all_features = self.model["encoder"](all_frames)
            all_features = [torch.split(feature, self.batch_size) for feature in all_features]

            features = {}
            for index, frame_id in enumerate(self.frame_ids):
                features[frame_id] = [feature[index] for feature in all_features]
            outputs = self.model["decoder"](features[0])

        elif self.pose_type == "separate": # separate 타입이면 posecnn처럼 이미지 한 장을 포워드
            features = self.model["encoder"](inputs[("color_aug", 0, 0)])
            outputs  = self.model["decoder"](features)

        """
        1. color_aug를 으로 추출한 outputs에 disparity뿐만 아니라 R, T, cam2cam 키를 추가
        2. train_inputs, train_outputs를 이용해서 [R|T]를 계산하고 camera_coords -> frame_coords으로 변환
        3. outputs에 스케일마다 인터폴레이트한 뎁스 저장, 
           스케일마다 프레임 코디네이트 저장,
           스케일마다 그리드 샘플링한 이미지 저장 (소스 이미지를 타겟 좌표 관점에서 바라본 것)
        """
        inputs, outputs = self.compute_pose(inputs, outputs, features)
        inputs, outputs = self.compute_depth(inputs, outputs)
        loss            = self.compute_loss(inputs, outputs)
        return outputs, loss


    def compute_pose(self, inputs, outputs, features): # 포즈 계산
        """
        포즈를 계산하는 함수
        프레임 아이디가 [0, -2, -1, 1, 2] 이면 [-2, -1, 1, 2]를 순회해서
        [-2, 0], [-1, 0], [0, 1], [0, 2] 형태로 묶어, backward pose, forward pose를 추정
        framd_id < 0 -> 0이 뒤에 오게 묶고, frame_id < 0이므로 invert matrix = True
        frame_id > 0 -> 0이 앞에 오게 묶고, frame_id > 0이므로 invert matrix = False
        """
        if self.num_pose_frames == 2:
            if self.pose_type == "posecnn":
                all_frames = {frame_id: inputs[("color_aug", frame_id, 0)] for frame_id in self.frame_ids}

                for frame_id in self.frame_ids[1:]:
                    if frame_id < 0:
                        pose_inputs = torch.cat([all_frames[frame_id], all_frames[0]], dim = 1)
                    else:
                        pose_inputs = torch.cat([all_frames[0], all_frames[frame_id]], dim = 1)
                    
                    axisangle, translation        = self.model["pose_decoder"](pose_inputs)
                    outputs[("R", frame_id, 0)]   = axisangle
                    outputs[("T", frame_id, 0)]   = translation
                    outputs[("c2c", frame_id, 0)] = param2matrix(
                        axisangle = axisangle[:, 0], translation = translation[:, 0], invert = (frame_id < 0))

            elif self.pose_type == "shared":
                all_features = {frame_id: features[frame_id] for frame_id in self.frame_ids}

                for frame_id in self.frame_ids[1:]:
                    if frame_id < 0:
                        pose_inputs = [all_features[frame_id], all_features[0]]
                    else:
                        pose_inputs = [all_features[0], all_features[frame_id]]

                    axisangle, translation        = self.model["pose_decoder"](pose_inputs)
                    outputs[("R", frame_id, 0)]   = axisangle
                    outputs[("T", frame_id, 0)]   = translation
                    outputs[("c2c", frame_id, 0)] = param2matrix(
                        axisangle = axisangle[:, 0], translation = translation[:, 0], invert = (frame_id < 0))

            elif self.pose_type == "separate":
                all_frames = {frame_id: inputs[("color_aug", frame_id, 0)] for frame_id in self.frame_ids}

                for frame_id in self.frame_ids[1:]:
                    if frame_id < 0:
                        pose_inputs = torch.cat([all_frames[frame_id], all_frames[0]], dim = 1)
                    else:
                        pose_inputs = torch.cat([all_frames[0], all_frames[frame_id]], dim = 1)

                    pose_inputs = [self.model["pose_encoder"](pose_inputs)]
                    axisangle, translation = self.model["pose_decoder"](pose_inputs)
                    outputs[("R", frame_id, 0)]   = axisangle
                    outputs[("T", frame_id, 0)]   = translation
                    outputs[("c2c", frame_id, 0)] = param2matrix(
                        axisangle = axisangle[:, 0], translation = translation[:, 0], invert = (frame_id < 0))
                    
        else:
            if self.pose_type == "posecnn":
                all_frames             = torch.cat([inputs[("color_aug", frame_id, 0)] for frame_id in self.frame_ids], dim =1)
                axisangle, translation = self.model["pose_decoder"](all_frames)
            elif self.pose_type == "shared":
                all_features           = [features[frame_id] for frame_id in self.frame_ids]
                axisangle, translation = self.model["pose_decoder"](all_features)
            elif self.pose_type == "separate":
                all_frames             = torch.cat([inputs[("color_aug", frame_id, 0)] for frame_id in self.frame_ids], dim = 1)
                pose_inputs            = [self.model["pose_encoder"](all_frames)]
                axisangle, translation = self.model["pose_decoder"](pose_inputs)
            
            for index, frame_id in enumerate(self.frame_ids[1: ]):
                outputs[("R", frame_id, 0)]   = axisangle
                outputs[("T", frame_id, 0)]   = translation
                outputs[("c2c", frame_id, 0)] = param2matrix(
                    axisangle = axisangle[:, index], translation = translation[:, index])
        return inputs, outputs  # outputs에 "R", "T", "cam2cam" 키를 가지고 나옴
        

    def compute_depth(self, inputs, outputs): # 뎁스 계산
        """
        1. 추정한 pose와 disparity로 depth 및 warping image를 계산 (default_scale = 0 -> (320, 1024)를 의미)
        2. 4개 스케일의 디스패리티 맵을 default_scale (원래 이미지 크기) 로 interpolate
        3. 원본 크기의 disparity를 얻은 후, 동일 크기의 뎁스로 변환하여 로스에 사용
        4. 키 프레임을 제외한 warping image와 target -> photometric loss
        5. 4개의 이미지 스케일과 디스패리티 스케일 -> smooth loss
        """
        default_scale = 0
        for scale in self.scales:
            disparity = outputs[("disp", scale)]
            disparity = F.interpolate(
                            disparity, [self.height, self.width], mode = "bilinear", align_corners = False)
            _, depth  = disparity2depth(disparity, self.min_depth, self.max_depth)
            outputs[("depth", 0, scale)] = depth # saving depth for depth eval

            for frame_id in self.frame_ids[1: ]: # 키 프레임 제외하고 [-2, -1, 1, 2] or [-1 ,1] etc ...
                """
                PoseCNN은 https://arxiv.org/abs/1712.00175가 출처
                shared는 monodepth2의 arxiv ver1
                separate가 monodepth2의 arxiv ver2

                Ps ~ KTDK^-1Pt
                0. R, T를 이용하여 transformation 행렬 계산
                1. frame_id = 0에서 default_scale의 뎁스 정보를 이용하여 카메라 좌표 획득
                   K, transformation으로 다른 frame_id ([-2, -1, 1, 2]) 뷰들로 변환
                2. id = 0의 프레임에서 id = frame_id의 프레임으로 옮긴 좌표를 저장
                3. 그 좌표와 해당 frame_id 이미지로 grid_sample을 통한 warping,
                   warping 이미지를 output[("warped_color" ~~ )]에 저장
                """
                if self.pose_type == "posecnn": # https://arxiv.org/abs/1712.00175
                    axisangle      = outputs[("R", frame_id, default_scale)]
                    translation    = outputs[("T", frame_id, default_scale)]
                    inv_depth      = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)
                    transformation = param2matrix(
                                        axisangle = axisangle[:, 0], 
                                        translation = translation[:, 0] * mean_inv_depth[:, 0], 
                                        invert = (frame_id < 0))
                else:
                    transformation = outputs[("c2c", frame_id, default_scale)]
                                    

                camera_coords  = self.backward_project(depth, inputs[("inv_K", default_scale)])
                frame_coords   = self.forward_project(camera_coords, inputs[("K", default_scale)], transformation)
                outputs[("warped_color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, default_scale)],
                    frame_coords,
                    padding_mode = "border",
                    align_corners = False)
        # outputs에 "depth", warped_color" 키 값들을 가지고 리턴
        return inputs, outputs


    def compute_loss(self, inputs, outputs): # 로스 계산
        """
        타겟 이미지, 와핑 이미지를 가지고 로스를 계산하는 함수
        1. 스케일 리스트를 돌면서 scale: 0, 1, 2, 3에 대한 뎁스 계산
        2. 해당 스케일의 디스패리티와 타겟 이미지, scale = 0의 원본 타겟 이미지 지정
        3. 프레임 아이디 리스트를 돌면서 reprojection_loss 계산 (와핑이 잘 되면 로스 0, 와핑이 잘 안되면 로스 큼)

        4. (매우 중요) self.opt.use_automasking을 켜서 no_reprojection_loss 계산
            reprojection_loss warping한 이미지와 target 사이의 로스를 계산
            no_reprojection_loss warping하지 않은 원본과 타겟을 계산 

            reprojection_loss < no_reprojection_loss
            포즈를 잘 알아서 와핑시킨 이미지가 타겟 이미지와 더욱 잘 일치하다는 것의 의미?
            정적인 물체가 배경처럼 간주됨, 그러니까 카메라와 같은 속도, 방향으로 움직이는 물체 -> 로스에서 무시

            reprojection_loss > no_reprojection_loss
            포즈를 잘 알아서 와핑시켰는데 오히려 와핑 시키지 않은 이미지와의 로스보다도 크다. 어떤 의미?
            포즈에 비해 동적인 물체 모션이 별도로 있다. 이것은 로스에 반영해야겠다.

        5. 이 둘을 하나로 합치고, 나머지 smooth loss를 계산해서 더 함
        """
        loss_dict     = {}
        total_loss    = 0
        default_scale = 0
        for scale in self.scales: # 0 1 2 3
            scale_loss     = 0
            disparity      = outputs[("disp", scale)]             # 다양한 사이즈의 스케일 (4개)
            target_scale   = inputs[("color", 0, scale)]          # 다양한 사이즈의 스케일 (4개)
            target_default = inputs[("color", 0, default_scale)]   # scale = 0의 타겟 이미지
            
            # target과 warping 이미지에 대해서 reprojection_loss 계산
            reprojection_loss = []
            for frame_id in self.frame_ids[1: ]:
                prediction = outputs[("warped_color", frame_id, scale)]         # prediction: [B C 320 1024]
                warp_loss  = self.reprojection_loss(prediction, target_default)   # depth: [B C 320 1024]
                reprojection_loss.append(warp_loss)
            reprojection_loss = torch.cat(reprojection_loss, 1)
            
            # if self.opt.use_automasking(True) == if not self.disable_automasking(False)
            # 원본 이미지와 타겟 이미지의 reprojection loss 계산 -> 변하지 않는 부분은 로스가 작을 것
            if self.opt.use_automasking:
                no_reprojection_loss = []
                for frame_id in self.frame_ids[1: ]:
                    source_default  = inputs[("color", frame_id, default_scale)]
                    no_warping_loss = self.reprojection_loss(source_default, target_default)
                    no_reprojection_loss.append(no_warping_loss)
                no_reprojection_loss = torch.cat(no_reprojection_loss, 1)
            
            if self.opt.use_automasking:
                no_reprojection_loss += 0.00001 * torch.randn(no_reprojection_loss.shape).to(self.device)
                combined_loss        = torch.cat((no_reprojection_loss, reprojection_loss), dim = 1)
            else:
                combined_loss        = reprojection_loss


            if combined_loss.shape[1] == 1:
                selected_reproejction = combined_loss
            else:
                selected_reproejction, _ = torch.min(combined_loss, dim = 1)

            # disparity와 color로 smooth loss 계산
            smooth_loss = self.smooth_loss(disp = disparity, color = target_scale)
            
            # Loss summation
            scale_loss = scale_loss + selected_reproejction.mean()
            scale_loss = scale_loss + self.opt.disparity_smoothness * smooth_loss / (2**scale)
            total_loss = total_loss + scale_loss
        
        total_loss        = total_loss / self.num_scales
        loss_dict["loss"] = total_loss
        return loss_dict


    def compute_metric(self, inputs, outputs, loss, log_dict): # 배치 데이터마다 메트릭 계산
        log_dict["loss"].append(loss["loss"].detach().cpu().numpy())

        depth_errors = compute_depth_metric(inputs, outputs, "torch")
        for index, metric in enumerate(self.depth_losess_name[1:]):
            log_dict[metric].append(depth_errors[index].cpu().numpy())
        return log_dict


    def set_train(self): # 훈련 모드
        for m in self.model.values():
            m.train()

    def set_valid(self): # 검증 모드
        for m in self.model.values():
            m.eval()


if __name__ == "__main__":
    args    = main()
    TRAINER = trainer(options = args)
    TRAINER.model_train()

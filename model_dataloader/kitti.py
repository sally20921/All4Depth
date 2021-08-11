import os
import copy
import random

import cv2
import numpy as np
from PIL import Image # using pillow-simd for increased speed
import torch
from torchvision import transforms
from torch.utils.data import Dataset

import skimage.transform
from albumentations import Resize
from albumentations.pytorch.transforms import ToTensor
from albumentations.augmentations.transforms import HorizontalFlip
from albumentations.augmentations.transforms import ColorJitter
from model_utility import *



class GetKITTI(object):
    def __init__(self, datapath: str, mode: list, cut: list):
        """
        KITTI 데이터 폴더의 구조
        ㄴKITTI
            ㄴ2011_09_26/2011_09_26_drive_0001_sync
                    ㄴimage_00/data
                    ㄴimage_01/data
                    ㄴimage_02/data
                        ㄴ0000000000.jpg
                        ㄴ0000000001.jpg
                    ㄴimage_03/data
                        ㄴ0000000000.jpg
                        ㄴ0000000001.jpg
                    ㄴvelodyne_points/data
                        ㄴ0000000000.bin
                        ㄴ0000000001.bin
            ㄴ2011_09_26/2011_09_26_drive_0002_sync
            ㄴ2011_09_28
            ㄴ2011_09_29
            ㄴ2011_09_30
            ㄴ2011_10_03
        
        벨로다인 포인트 파일을 기준으로 키프레임 선택 (키프레임은 GT가 존재해야함)
        따라서 벨로다인 파일의 확장자를 제거한 프레임 인덱스로 키프레임 형식으로 지정
        해당 프레임 인덱스 이름에 yyyy_mm_dd 형태와 side_map을 섞어서 부여
        (레거시 kitti_splits 규약을 따름)

        ex)
        2011_10_03/2011_10_03_drive_0034_sync 0000000190 r
        2011_10_03/2011_10_03_drive_0034_sync 0000000307 l
        2011_09_26/2011_09_26_drive_0117_sync 0000000134 r
        2011_09_30/2011_09_30_drive_0028_sync 0000000534 l

        Args:
            datapath: "./dataset/kitti"
            model: "train" or "val" 데이터 폴더 리스트
            side: "l" or "r"
        """
        self.datapath  = datapath
        self.mode      = mode
        self.cut       = cut
        self.side_map  = {"2": 2, "3": 3, "l": 2, "r": 3}
        self.l_path    = "image_02/data"
        self.r_path    = "image_03/data"
        self.velo_path = "velodyne_points/data"


    def search(self):
        all_filename = []
        for yyyy_mm_dd in self.mode: # 각 날짜 별 폴더마다 순회
            velodyne_path = os.path.join(self.datapath, yyyy_mm_dd, self.velo_path)
            # left_path  = os.path.join(self.datapath, yyyy_mm_dd, self.l_path)
            # right_path = os.path.join(self.datapath, yyyy_mm_dd, self.r_path)

            # 왼쪽 카메라 파일 (image_02) 와 오른쪽 카메라 파일 (image_03)을 순회
            left_filename  = []
            right_filename = []
            for filename in os.listdir(velodyne_path): # 벨로다인 파일 이름을 순회
                # 0000000011.bin, ... ... 0000000035.bin 파일을 0000000011, 0000000035으로 쪼갬
                frame_index = filename.split(".bin")[0]

                # 쪼갠 frame_index은 image_02, image_03 폴더에 이미지 파일로 매칭됨 (스테레오 이미지)
                left_filename.append("".join([yyyy_mm_dd, " ", frame_index, " ", "l"]))
                right_filename.append("".join([yyyy_mm_dd, " ", frame_index, " ", "r"]))
            
            # 상대적인 프레임 아이디를 위해 양 끝 N개는 잘라줌
            modified_left_filename  = self.side_cut(left_filename, self.cut)
            modified_right_filename = self.side_cut(right_filename, self.cut)
            all_filename += modified_left_filename
            all_filename += modified_right_filename

        print("전체 길이  :  {}".format(len(all_filename)))
        random.shuffle(all_filename)
        return all_filename


    def side_cut(self, filename, cut):
        """
        이 클래스가 존재하는 이유
        프레임 인덱스가 키 프레임 기준으로 좌, 우 몇 까지 consecutive frame을 쓸 지 모름
        만약 키 프레임 인덱스가 0이면 왼쪽 consecutive frame은 음수가 되기 때문에 사용할 수 없음
        그래서 consecutive frame 범위를 두기 위해 맨 끝 프레임은 잘라내는 목적
        """        
        modified_filename = []
        modified_filename = filename[cut[0]: len(filename)-cut[1]]
        return modified_filename



class KITTIMonoDataset(Dataset):
    def __init__(self, datapath, filename, is_training, frame_ids, ext, scale = 1):
        super(KITTIMonoDataset, self).__init__()
        """
        Args:
            datapath: "./dataset/kitti"
            filename: splits file of KITTI
            is_training: True or False
            frame_ids: relative position list of key frame
            mode: "train" or "val" or "test"
            ext: ".jpg" or ".png"
            scale: 1
        
        interpolation 1은 쓰지 말 것, 성능이 나오지 않음, 0 아니면 3으로 실험
        albumentation Resize interpolation option
        0 : cv2.INTER_NEAREST, 
        1 : cv2.INTER_LINEAR, 
        2 : cv2.INTER_CUBIC, 
        3 : cv2.INTER_AREA, 
        4 : cv2.INTER_LANCZOS4. Default: cv2.INTER_LINEAR.
        """
        self.datapath     = datapath
        self.filename     = filename
        self.is_training  = is_training
        self.frame_ids    = frame_ids
        self.ext          = ext
        self.scale        = scale
        self.inter        = cv2.INTER_AREA # Image.ANTIALIAS와 동등한가?
        self.side_map     = {"2": 2, "3": 3, "l": 2, "r": 3}
        # intrinsic camera (https://github.com/nianticlabs/monodepth2)
        self.K            = np.array([[0.58,    0,    0.5,    0],
                                      [0,    1.92,    0.5,    0],
                                      [0,       0,      1,    0],
                                      [0,       0,      0,    1]], dtype = np.float32) 
        self.resize        = {}
        self.scales        = list(range(4))
        self.origin_scale  = (375, 1242)
        self.default_scale = [(320, 1024), (192, 640), (96, 320), (48, 160)] # corresponding as scale: 0, 1, 2, 3
        self.scale_list    = [(self.default_scale[self.scale][0]//(2**i), 
                               self.default_scale[self.scale][1]//(2**i)) for i in self.scales]

        """
        데이터로더 프로세스 플로우
        1. 좌우로 뒤집을지 말지 결정하는 do_flip(bool)과 함께 이미지를 로드 (원본 이미지)
           키티 데이터 원본 크기는 (375, 1245)
        2. 원본 스케일 이미지를 원하는 스케일로 바꾸고, 그 스케일부터 2배율로 줄어드는 리스케일 [0, 1, 2, 3]
           ("color", <frame_id>, <scale>) 키 형태로 저장
        3. is_training 모드이면 do_auge를 주고, 각 이미지마다 augmentation을 적용
           ("color_aug", <frame_id>, <scale>) 키 형태로 저장
        4. GT로 사용하는 Point2Depth 이미지는 원본 스케일로 저장
           최소 스케일에 맞게 세팅된 K는 monodepth2의 intrinsic parameter를 따름 
           -> 변형한 이미지 스케일을 곱해서 복원
        5. 마지막으로 input_data의 모든 키 값들을 numpy2tensor 변환
        """
        for scale, (height, width) in enumerate(self.scale_list):
            self.resize[scale] = Resize(
                height = int(height), width  = int(width), interpolation = self.inter)
        self.depth_resize   = Resize(
            height = self.origin_scale[0], width = self.origin_scale[1], interpolation = 0)

        self.augment_key    = "image"
        self.brightness     = (0.8, 1.2)
        self.contrast       = (0.8, 1.2)
        self.saturation     = (0.8, 1.2)
        self.hue            = (-0.1, 0.1)
        self.HorizontalFlip = HorizontalFlip(p = 1.0)
        self.ColorJitter    = ColorJitter(
            brightness = self.brightness, contrast = self.contrast, saturation = self.saturation, hue = self.hue, p = 1.0)
        self.image2tensor   = ToTensor()
        print("-- KITTI scaling table")
        print("interpolation     :  {0}".format(self.inter))
        print("Scale factor      :  {0}".format(self.scale))
        print("Default 0 scale   :  {0} {1}".format(self.default_scale[0][0], self.default_scale[0][1]))
        print("Default 1 scale   :  {0} {1}".format(int(self.default_scale[1][0]), int(self.default_scale[1][1])))
        print("Default 2 scale   :  {0} {1}".format(int(self.default_scale[2][0]), int(self.default_scale[1][1])))
        print("Default 3 scale   :  {0} {1}".format(int(self.default_scale[3][0]), int(self.default_scale[1][1])))
        print("Resolution List   :  {0}".format(self.scale_list))


    def get_image_path(self, folder_name, frame_index, side):
        """
        문자열 앞에 0을 붙이는 여러가지 방법
        "{:010d}{}".format(index, ext)
        f"{index:010d}" + ext
        """
        image_name = "{:010d}{}".format(frame_index, self.ext)
        image_path = os.path.join(
            self.datapath, folder_name , "image_0{}/data/{}".format(self.side_map[side], image_name))
        return image_path

    def get_point_path(self, folder_name, key_frame):
        calib_path = os.path.join(self.datapath, folder_name.split("/")[0])
        point_name = "velodyne_points/data/{:010d}.bin".format(key_frame)
        point_path = os.path.join(self.datapath, folder_name, point_name)
        return calib_path, point_path


    def load_image(self, image_path, do_flip): # 이미지를 로드, 나중에 PIL로 고치기
        with open(image_path, 'rb') as f:
            with Image.open(f) as img:
                image_instance = img.convert('RGB')
                numpy_image    = np.array(image_instance)

                if do_flip == True:
                    numpy_image = self.flip_image(numpy_image)
                return numpy_image

    def load_point(self, calib_path, point_path, side, do_flip):
        """
        키 프레임의 포인트 클라우드를 불러오고, 원본 스케일로 리사이즈하여 input_data 뎁스 키에 저장
        Args:
            input_data: 데이터를 담을 딕셔너리
            index:      배치로 들어갈 데이터의 인덱스
            do_flip:    뒤집을꺼면 미리 뒤집자 (is_training 여부에서 미리 결정)

        이슈)
        문제: 계속해서 결과물의 시각적 느낌과는 다르게, 뎁스 메트릭이 정확하게 측정 안되는 문제 발생
        원인: 포인트클라우드에서 추출한 뎁스 맵을 전처리할 때 잘못 전처리 된 듯함 
        --> 향후 어떤 차이가 있는지 잘 파악할 것

        해결 1
        skimage.transform.resize 함수로 해결 (monodepth2의 방식을 그대로 따름)
        order = 0, presevr_range = True, mode = "constant" 인자가 무엇인지 이해할 것
        
        해결 2
        Albumentations.Resize 메서드를 그대로 사용하되, interpolation = 0으로 둘 것
        
        해결 3
        포인트 클라우드 좌표가 몇 번 카메라로 매핑될지 결정해야함, 기본 값은 cam = 2이였는데
        3번 카메라를 쓰면 3번 카메라 좌표로 변환되어야 하지만 메트릭이 정확해지지 않음
        """
        depth = Point2Depth(
            calib_path = calib_path, point_path = point_path, cam = self.side_map[side], vel_depth = False)
        depth = skimage.transform.resize(
                    depth, (1242, 375)[::-1], order = 0, preserve_range = True, mode = "constant")
        depth = np.reshape(depth, (depth.shape[0], depth.shape[1], 1))
        # depth = self.depth_resize(image = depth)
        # depth = np.reshape(depth["image"], (depth["image"].shape[0], depth["image"].shape[1], 1))
        
        if do_flip == True:
            depth = self.flip_image(depth)
        return depth


    def flip_image(self, numpy_image):
        numpy_image = self.HorizontalFlip(image = numpy_image)
        return numpy_image[self.augment_key]

    def resize_image(self, scale, numpy_image):
        numpy_image = self.resize[scale](image = numpy_image)
        return numpy_image[self.augment_key]

    def recolor_image(self, numpy_image):
        numpy_image = self.ColorJitter(image = numpy_image)
        return numpy_image[self.augment_key]

    def numpy2tensor(self, numpy_image):
        tensor_image = self.image2tensor(image = numpy_image)
        return tensor_image[self.augment_key]


    def preprocessing_image(self, input_data, folder_name, key_frame, side, do_flip):
        """
        key_frame는 키프레임 (시퀀스의 중앙에 있을수도, 맨 뒤에 있을수도 있음)
        frame_ids가 중요한데 key_frame (키 프레임) 기준으로 상대적인 위치를 나타냄

        ex)
        key_frame = 123, frame_ids = [-1, 0, 1]
        for index in frame_ids:
            outputs = load_image(index + key_frame)
        """
        for frame_id in self.frame_ids:
            image_path  = self.get_image_path(folder_name, frame_id + key_frame, side)
            image_array = self.load_image(image_path, do_flip)
            input_data.update({("color", frame_id, scale): self.resize_image(scale, image_array) for scale in self.scales})
        return input_data

    def preprocessing_point(self, input_data, folder_name, key_frame, side, do_flip):
        """
        1. 캘리브레이션, 포인트 클라우드 파일 로드
        2. 포인트 클라우드 -> 뎁스 이미지로 변환한 데이터를 로드
        3. input_data 딕셔너리에 키 프레임의 뎁스 이미지 데이터 저장
        """
        calib_path, point_path   = self.get_point_path(folder_name, key_frame)
        depth                    = self.load_point(calib_path, point_path, side, do_flip)
        input_data[("depth", 0)] = depth.astype(np.float32)
        return input_data

    def preprocessing_intrinsic(self, input_data):
        """
        1. 원본 intrinsic을 사용할 경우, "스케일링 크기 / 원본 크기" 비율을 곱해서 intrinsic을 줄여줌
        2. monodepth2의 intrinsic을 사용할 경우, 스케일링 크기만 곱해서 intrinsic을 늘려줌
        """
        for scale in self.scales:
            K_copy       = self.K.copy()
            K_copy[0, :] = K_copy[0, :] * self.scale_list[scale][1]
            K_copy[1, :] = K_copy[1, :] * self.scale_list[scale][0]
            inv_K        = np.linalg.pinv(K_copy)

            input_data[("K", scale)]     = torch.from_numpy(K_copy)
            input_data[("inv_K", scale)] = torch.from_numpy(inv_K)
        return input_data


    def __getitem__(self, index):
        """
        returns 
            ("color", <frame_id>, <scale>)             for raw color images,
            ("color_aug", <frame_id>, <scale>)         for aug color images,
            ("depth", 0) and ("K", 0) and ("inv_K", 0) for depth, intrinsic of key frame
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        do_flip     = self.is_training and random.random() > 0.5
        do_auge     = self.is_training and random.random() > 0.5
        
        batch_line  = self.filename[index].split()
        folder_name = batch_line[0]
        key_frame   = int(batch_line[1])
        side        = batch_line[2]
    
        # input_data 딕셔너리를 지정하고, folder_name, key_frame, side 여부,
        # do_flip으로 이미지 전처리와 뎁스 전처리
        input_data = {}
        input_data = self.preprocessing_image(input_data, folder_name, key_frame, side, do_flip)        
        input_data = self.preprocessing_point(input_data, folder_name, key_frame, side, do_flip)

        if do_auge:
            for frame_id in self.frame_ids:
                input_data.update({("color_aug", frame_id, scale):
                    self.recolor_image(input_data[("color", frame_id, scale)]) for scale in self.scales})
        else:
            for frame_id in self.frame_ids:
                input_data.update({("color_aug", frame_id, scale):
                    input_data[("color", frame_id, scale)] for scale in self.scales})
        
        # input_data에 포함된 모든 키의 값을 torch.tensor 타입으로 변환
        input_data.update({key: self.numpy2tensor(input_data[key]) for key in input_data})

        # 원본 이미지를 스케일링한 비율만큼 K, inv_K도 동일하게 스케일링
        input_data = self.preprocessing_intrinsic(input_data)
        return input_data

    def __len__(self):
        return len(self.filename)



###################################################################################################################
###################################################################################################################
###################################################################################################################


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(Dataset):
    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg'):
        super(MonoDataset, self).__init__()
        """
        Superclass for monocular dataloaders

        Args:
            data_path
            filenames
            height
            width
            frame_idxs
            num_scales
            is_train
            img_ext
        """
        self.data_path  = data_path
        self.filenames  = filenames
        self.height     = height
        self.width      = width
        self.num_scales = num_scales
        self.interp     = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train   = is_train
        self.img_ext    = img_ext

        self.loader     = pil_loader
        self.to_tensor  = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast   = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue        = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast   = 0.2
            self.saturation = 0.2
            self.hue        = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize(
                (self.height // s, self.width // s), interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """
        Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """
        Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            ("depth", 0)                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip      = self.is_train and random.random() > 0.5

        line   = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs[("depth", 0)] = np.expand_dims(depth_gt, 0)
            inputs[("depth", 0)] = torch.from_numpy(inputs[("depth", 0)].astype(np.float32))

        # if "s" in self.frame_idxs:
        #     stereo_T = np.eye(4, dtype=np.float32)
        #     baseline_sign = -1 if do_flip else 1
        #     side_sign = -1 if side == "l" else 1
        #     stereo_T[0, 3] = side_sign * baseline_sign * 0.1
        #     inputs["stereo_T"] = torch.from_numpy(stereo_T)
        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError



class KITTIDataset(MonoDataset):
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)
        """
        Superclass for different types of KITTI dataset loaders
        NOTE: Make sure your intrinsics matrix is *normalized* by the original image size.
        To normalize you need to scale the first row by 1 / image_width and the second row
        by 1 / image_height. Monodepth2 assumes a principal point to be exactly centered.
        If your principal point is far from the center you might need to disable the horizontal
        flip augmentation.
        """
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (1242, 375)
        self.side_map       = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line        = self.filenames[0].split()
        scene_name  = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path, scene_name, "velodyne_points/data/{:010d}.bin".format(int(frame_index)))
        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(Image.open.FLIP_LEFT_RIGHT)
        return color



class KITTIRAWDataset(KITTIDataset):
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)
        """
        KITTI dataset which loads the original velodyne depth maps for ground truth
        """
    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path, folder, "velodyne_points/data/{:010d}.bin".format(int(frame_index)))
        depth_gt = Point2Depth(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order = 0, preserve_range = True, mode = 'constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt



class KITTIOdomDataset(KITTIDataset):
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)
        """
        KITTI dataset for odometry training and testing
        """
    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path



class KITTIDepthDataset(KITTIDataset):
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)
        """
        KITTI dataset which uses the updated ground truth depth maps
        """
    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path, folder, "proj_depth/groundtruth/image_0{}".format(self.side_map[side]), f_str)

        depth_gt = Image.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, Image.open.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)
        return depth_gt
import copy
from pprint import pprint
import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from torchvision import transforms
from iso.data.utils.helpers import (
    vox2pix,
    compute_local_frustums,
    compute_CP_mega_matrix,
)
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt

import cv2


class OccScanNetDataset(Dataset):
    def __init__(
        self,
        split,
        interval=-1,
        train_scenes_sample=-1,
        val_scenes_sample=-1,
        n_relations=4,
        color_jitter=None,
        frustum_size=4,
        fliplr=0.0,
        v2=False,
    ):  
        # cur_dir = os.path.abspath(os.path.curdir)
        iso_mm_path = os.getenv["ISO_MM_PATH"]
        
        self.n_relations = n_relations
        self.frustum_size = frustum_size
        self.split = split
        self.fliplr = fliplr
        
        self.voxel_size = 0.08  # 0.08m
        self.scene_size = (4.8, 4.8, 2.88)  # (4.8m, 4.8m, 2.88m)
        
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        
        # print(os.getcwd())
        if v2:
            subscenes_list = f'{iso_mm_path}/data/occscannet/{self.split}_subscenes_v2.txt'
        else:  # data/occscannet/train_subscenes.txt
            subscenes_list = f'{iso_mm_path}/data/occscannet/{self.split}_subscenes.txt'
        with open(subscenes_list, 'r') as f:
            self.used_subscenes = f.readlines()
            for i in range(len(self.used_subscenes)):
                self.used_subscenes[i] = f'{iso_mm_path}/data/occscannet/' + self.used_subscenes[i].strip()
       
        if "train" in self.split:
            # breakpoint()
            if train_scenes_sample != -1:
                self.used_subscenes = self.used_subscenes[:train_scenes_sample]
            # if interval != -1:
            #     self.used_subscenes = self.used_subscenes[::interval]
            # print(f"Total train scenes number: {len(self.used_scan_names)}")
            # pprint(self.used_subscenes)
            print(f"Total train scenes number: {len(self.used_subscenes)}")
        elif "val" in self.split:
            # print(f"Total validation scenes number: {len(self.used_scan_names)}")
            # pprint(self.used_subscenes)
            if val_scenes_sample != -1:
                self.used_subscenes = self.used_subscenes[:val_scenes_sample]
            print(f"Total validation scenes number: {len(self.used_subscenes)}")

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        # os.chdir(cur_dir)

    def __getitem__(self, index):
        name = self.used_subscenes[index]
        iso_mm_path = os.getenv["ISO_MM_PATH"]
        name = f"{iso_mm_path}/"+name
        with open(name, 'rb') as f:
            data = pickle.load(f)

        cam_pose = data["cam_pose"]
        cam_intrin = data['intrinsic']
        
        img = cv2.imread(data['img'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data['img'] = img
        data['raw_img'] = copy.deepcopy(cv2.resize(img, (640, 480)))
        depth_img = Image.open(data['depth_gt']).convert('I;16')
        depth_img = np.array(depth_img) / 1000.0
        data['depth_gt'] = depth_img
        depth_gt = data['depth_gt']
        # print(depth_gt.shape)
        img_H, img_W = img.shape[0], img.shape[1]
        img = cv2.resize(img, (640, 480))
        # plt.imshow(img)
        # plt.savefig('img.jpg')
        W_factor = 640 / img_W
        H_factor = 480 / img_H
        img_H, img_W = img.shape[0], img.shape[1]
        
        cam_intrin[0, 0] *= W_factor
        cam_intrin[1, 1] *= H_factor
        cam_intrin[0, 2] *= W_factor
        cam_intrin[1, 2] *= H_factor
        
        data["cam_pose"] = cam_pose
        T_world_2_cam = np.linalg.inv(cam_pose)
        vox_origin = list(data["voxel_origin"])
        vox_origin = np.array(vox_origin)
        data["vox_origin"] = vox_origin
        data["cam_k"] = cam_intrin[:3, :3][None]
        
        
        target = data[
            "target_1_4"
        ]  # Following SSC literature, the output resolution on NYUv2 is set to 1:4
        target = np.where(target == 255, 0, target)
        data["target"] = target
        data["target_1_4"] = target
        target_1_4 = data["target_1_16"]
        target_1_4 = np.where(target_1_4 == 255, 0, target_1_4)
        data["target_1_16"] = target_1_4
        
        CP_mega_matrix = compute_CP_mega_matrix(
            target_1_4, is_binary=self.n_relations == 2
        )
         
        data["CP_mega_matrix"] = CP_mega_matrix

        # compute the 3D-2D mapping
        projected_pix, fov_mask, pix_z = vox2pix(
            T_world_2_cam,
            cam_intrin,
            vox_origin,
            self.voxel_size,
            img_W,
            img_H,
            self.scene_size,
        )
        # print(projected_pix)
        # print(fov_mask.shape)
        
        data["projected_pix_1"] = projected_pix
        data["fov_mask_1"] = fov_mask
        data['pix_z'] = pix_z
        
        # compute the masks, each indicates voxels inside a frustum
        
        frustums_masks, frustums_class_dists = compute_local_frustums(
            projected_pix,
            pix_z,
            target,
            img_W,
            img_H,
            dataset="OccScanNet",
            n_classes=12,
            size=self.frustum_size,
        )
        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        img = Image.fromarray(img).convert('RGB')
   
        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)

        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0

        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            # raw_img = np.ascontiguousarray(np.fliplr(raw_img))
            data['raw_img'] = np.ascontiguousarray(np.fliplr(data['raw_img']))
            data["projected_pix_1"][:, 0] = (
                img.shape[1] - 1 - data["projected_pix_1"][:, 0]
            )

            depth_gt = np.ascontiguousarray(np.fliplr(depth_gt))
            data['depth_gt'] = depth_gt
        # print(depth_gt.shape)

        data["img"] = self.normalize_rgb(img)  # (3, img_H, img_W)
        data["name"] = name

        return data
    
    def __len__(self):
        if 'train' in self.split:
            return len(self.used_subscenes)
        elif 'val' in self.split:
            return len(self.used_subscenes)

def test():
    datset = OccScanNetDataset(
        split='train',
        fliplr=0.5,
        frustum_size=8,
        color_jitter=(0.4, 0.4, 0.4),
    )
    datset = OccScanNetDataset(
        split='val',
        frustum_size=8,
        color_jitter=(0.4, 0.4, 0.4),
    )

if __name__ == "__main__":
    test()
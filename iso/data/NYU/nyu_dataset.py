import torch
import os
import glob
from torch.utils.data import Dataset
import numpy as np
from PIL import Image, ImageTransform
from torchvision import transforms
from iso.data.utils.helpers import (
    vox2pix,
    compute_local_frustums,
    compute_CP_mega_matrix,
)
import pickle
import torch.nn.functional as F
import copy


def read_depth(depth_path):
    depth_vis = Image.open(depth_path).convert('I;16')  
    depth_vis_array = np.array(depth_vis)

    arr1 = np.right_shift(depth_vis_array, 3)
    arr2 = np.left_shift(depth_vis_array, 13)
    depth_vis_array = np.bitwise_or(arr1, arr2)

    depth_inpaint = depth_vis_array.astype(np.float32) / 1000.0  
      
    return depth_inpaint


class NYUDataset(Dataset):
    def __init__(
        self,
        split,
        root,
        preprocess_root,
        n_relations=4,
        color_jitter=None,
        frustum_size=4,
        fliplr=0.0,
    ):
        self.n_relations = n_relations
        self.frustum_size = frustum_size
        self.n_classes = 12
        self.root = os.path.join(root, "NYU" + split)
        self.preprocess_root = preprocess_root
        self.base_dir = os.path.join(preprocess_root, "base", "NYU" + split)
        self.fliplr = fliplr

        self.voxel_size = 0.08  # 0.08m
        self.scene_size = (4.8, 4.8, 2.88)  # (4.8m, 4.8m, 2.88m)
        self.img_W = 640
        self.img_H = 480
        self.cam_k = np.array([[518.8579, 0, 320], [0, 518.8579, 240], [0, 0, 1]])

        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )

        self.scan_names = glob.glob(os.path.join(self.root, "*.bin"))

        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __getitem__(self, index):
        file_path = self.scan_names[index]
        filename = os.path.basename(file_path)
        name = filename[:-4]

        os.makedirs(self.base_dir, exist_ok=True)
        filepath = os.path.join(self.base_dir, name + ".pkl")

        with open(filepath, "rb") as handle:
            data = pickle.load(handle)

        cam_pose = data["cam_pose"]
        T_world_2_cam = np.linalg.inv(cam_pose)
        vox_origin = data["voxel_origin"]
        data["cam_k"] = self.cam_k
        target = data[
            "target_1_4"
        ]  # Following SSC literature, the output resolution on NYUv2 is set to 1:4
        data["target"] = target
        target_1_4 = data["target_1_16"]

        CP_mega_matrix = compute_CP_mega_matrix(
            target_1_4, is_binary=self.n_relations == 2
        )
        data["CP_mega_matrix"] = CP_mega_matrix

        # compute the 3D-2D mapping
        projected_pix, fov_mask, pix_z = vox2pix(
            T_world_2_cam,
            self.cam_k,
            vox_origin,
            self.voxel_size,
            self.img_W,
            self.img_H,
            self.scene_size,
        )
        
        data["projected_pix_1"] = projected_pix
        data["fov_mask_1"] = fov_mask

        # compute the masks, each indicates voxels inside a frustum
        frustums_masks, frustums_class_dists = compute_local_frustums(
            projected_pix,
            pix_z,
            target,
            self.img_W,
            self.img_H,
            dataset="NYU",
            n_classes=12,
            size=self.frustum_size,
        )
        data["frustums_masks"] = frustums_masks
        data["frustums_class_dists"] = frustums_class_dists

        rgb_path = os.path.join(self.root, name + "_color.jpg")
        depth_path = os.path.join(self.root, name + ".png")
        img = Image.open(rgb_path).convert("RGB")
        raw_img_pil = copy.deepcopy(img)
        raw_img = np.array(raw_img_pil)
        depth_gt = read_depth(depth_path)
        depth_gt = np.array(depth_gt, dtype=np.float32, copy=False)

        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0

        # randomly fliplr the image
        if np.random.rand() < self.fliplr:
            img = np.ascontiguousarray(np.fliplr(img))
            # raw_img = np.ascontiguousarray(np.fliplr(raw_img))
            raw_img_pil = raw_img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            data["projected_pix_1"][:, 0] = (
                img.shape[1] - 1 - data["projected_pix_1"][:, 0]
            )

            depth_gt = np.ascontiguousarray(np.fliplr(depth_gt))

        data["img"] = self.normalize_rgb(img)  # (3, img_H, img_W)
        data['depth_gt'] = depth_gt[None]  # (1, 480, 640)
        # data['raw_img'] = transforms.PILToTensor()(raw_img).float()
        # data['raw_img'] = transforms.ToTensor()(raw_img)
        data['raw_img'] = raw_img_pil
        # print(data['raw_img'].dtype)
        # print((data['raw_img'].numpy() - raw_img).max())
        data['pix_z'] = pix_z

        return data

    def __len__(self):
        return len(self.scan_names)

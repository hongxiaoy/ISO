import torch


def collate_fn(batch):
    data = {}
    imgs = []
    raw_imgs = []
    depth_gts = []
    targets = []
    names = []
    cam_poses = []

    pix_z = []

    vox_origins = []
    cam_ks = []

    CP_mega_matrices = []

    data["projected_pix_1"] = []
    data["fov_mask_1"] = []
    data["frustums_masks"] = []
    data["frustums_class_dists"] = []

    for idx, input_dict in enumerate(batch):
        CP_mega_matrices.append(torch.from_numpy(input_dict["CP_mega_matrix"]))
        for key in data:
            if key in input_dict:
                data[key].append(torch.from_numpy(input_dict[key]))

        cam_ks.append(torch.from_numpy(input_dict["cam_k"]).double())
        cam_poses.append(torch.from_numpy(input_dict["cam_pose"]).float())
        vox_origins.append(torch.from_numpy(input_dict["voxel_origin"]).double())

        pix_z.append(torch.from_numpy(input_dict['pix_z']).float())

        names.append(input_dict["name"])

        img = input_dict["img"]
        imgs.append(img)

        raw_img = input_dict['raw_img']
        raw_imgs.append(raw_img)

        depth_gt = torch.from_numpy(input_dict['depth_gt'])
        depth_gts.append(depth_gt)

        target = torch.from_numpy(input_dict["target"])
        targets.append(target)

    ret_data = {
        "CP_mega_matrices": CP_mega_matrices,
        "cam_pose": torch.stack(cam_poses),
        "cam_k": torch.stack(cam_ks),
        "vox_origin": torch.stack(vox_origins),
        "name": names,
        "img": torch.stack(imgs),
        # "raw_img": torch.stack(raw_imgs),
        "raw_img": raw_imgs,
        'depth_gt': torch.stack(depth_gts),
        "target": torch.stack(targets),
        'pix_z': torch.stack(pix_z),
    }
    for key in data:
        ret_data[key] = data[key]
    return ret_data

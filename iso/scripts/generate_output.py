from pytorch_lightning import Trainer
from iso.models.iso import ISO
from iso.data.NYU.nyu_dm import NYUDataModule
# from iso.data.semantic_kitti.kitti_dm import KittiDataModule
# from iso.data.kitti_360.kitti_360_dm import Kitti360DataModule
import hydra
from omegaconf import DictConfig
import torch
import numpy as np
import os
from hydra.utils import get_original_cwd
from tqdm import tqdm
import pickle


@hydra.main(config_name="../config/iso.yaml")
def main(config: DictConfig):
    torch.set_grad_enabled(False)

    # Setup dataloader
    if config.dataset == "kitti" or config.dataset == "kitti_360":
        feature = 64
        project_scale = 2
        full_scene_size = (256, 256, 32)

        if config.dataset == "kitti":
            data_module = KittiDataModule(
                root=config.kitti_root,
                preprocess_root=config.kitti_preprocess_root,
                frustum_size=config.frustum_size,
                batch_size=int(config.batch_size / config.n_gpus),
                num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            )
            data_module.setup()
            data_loader = data_module.val_dataloader()
            # data_loader = data_module.test_dataloader() # use this if you want to infer on test set
        else:
            data_module = Kitti360DataModule(
                root=config.kitti_360_root,
                sequences=[config.kitti_360_sequence],
                n_scans=2000,
                batch_size=1,
                num_workers=3,
            )
            data_module.setup()
            data_loader = data_module.dataloader()

    elif config.dataset == "NYU":
        project_scale = 1
        feature = 200
        full_scene_size = (60, 36, 60)
        data_module = NYUDataModule(
            root=config.NYU_root,
            preprocess_root=config.NYU_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
        )
        data_module.setup()
        data_loader = data_module.val_dataloader()
        # data_loader = data_module.test_dataloader() # use this if you want to infer on test set
    else:
        print("dataset not support")

    # Load pretrained models
    if config.dataset == "NYU":
        model_path = os.path.join(
            get_original_cwd(), "trained_models", "iso_nyu.ckpt"
        )
    else:
        model_path = os.path.join(
            get_original_cwd(), "trained_models", "iso_kitti.ckpt"
        )
    
    voxeldepth_res = []
    if config.voxeldepth:
        if config.voxeldepthcfg.depth_scale_1:
            voxeldepth_res.append('1')
        if config.voxeldepthcfg.depth_scale_2:
            voxeldepth_res.append('2')
        if config.voxeldepthcfg.depth_scale_4:
            voxeldepth_res.append('4')
        if config.voxeldepthcfg.depth_scale_8:
            voxeldepth_res.append('8')
    
    os.chdir(hydra.utils.get_original_cwd())

    model = ISO.load_from_checkpoint(
        model_path,
        feature=feature,
        project_scale=project_scale,
        fp_loss=config.fp_loss,
        full_scene_size=full_scene_size,
        voxeldepth=config.voxeldepth,
        voxeldepth_res=voxeldepth_res,
        #
        use_gt_depth=config.use_gt_depth,
        add_fusion=config.add_fusion,
        use_zoedepth=config.use_zoedepth,
        use_depthanything=config.use_depthanything,
        zoedepth_as_gt=config.zoedepth_as_gt,
        depthanything_as_gt=config.depthanything_as_gt,
        frozen_encoder=config.frozen_encoder,
    )
    model.cuda()
    model.eval()

    # Save prediction and additional data 
    # to draw the viewing frustum and remove scene outside the room for NYUv2
    output_path = os.path.join(config.output_path, config.dataset)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch["img"] = batch["img"].cuda()
            pred = model(batch)
            y_pred = torch.softmax(pred["ssc_logit"], dim=1).detach().cpu().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            for i in range(config.batch_size):
                out_dict = {"y_pred": y_pred[i].astype(np.uint16)}
                if "target" in batch:
                    out_dict["target"] = (
                        batch["target"][i].detach().cpu().numpy().astype(np.uint16)
                    )

                if config.dataset == "NYU":
                    write_path = output_path
                    filepath = os.path.join(write_path, batch["name"][i] + ".pkl")
                    out_dict["cam_pose"] = batch["cam_pose"][i].detach().cpu().numpy()
                    out_dict["vox_origin"] = (
                        batch["vox_origin"][i].detach().cpu().numpy()
                    )
                else:
                    write_path = os.path.join(output_path, batch["sequence"][i])
                    filepath = os.path.join(write_path, batch["frame_id"][i] + ".pkl")
                    out_dict["fov_mask_1"] = (
                        batch["fov_mask_1"][i].detach().cpu().numpy()
                    )
                    out_dict["cam_k"] = batch["cam_k"][i].detach().cpu().numpy()
                    out_dict["T_velo_2_cam"] = (
                        batch["T_velo_2_cam"][i].detach().cpu().numpy()
                    )

                os.makedirs(write_path, exist_ok=True)
                with open(filepath, "wb") as handle:
                    pickle.dump(out_dict, handle)
                    print("wrote to", filepath)


if __name__ == "__main__":
    main()

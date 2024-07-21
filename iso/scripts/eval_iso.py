from pytorch_lightning import Trainer
from iso.models.iso import ISO
from iso.data.NYU.nyu_dm import NYUDataModule
import hydra
from omegaconf import DictConfig
import torch
import os
from hydra.utils import get_original_cwd
from pytorch_lightning import seed_everything

torch.set_float32_matmul_precision('high')


@hydra.main(config_name="../config/iso.yaml")
def main(config: DictConfig):
    torch.set_grad_enabled(False)
    if config.dataset == "kitti":
        config.batch_size = 1
        n_classes = 20
        feature = 64
        project_scale = 2
        full_scene_size = (256, 256, 32)
        data_module = KittiDataModule(
            root=config.kitti_root,
            preprocess_root=config.kitti_preprocess_root,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
        )

    elif config.dataset == "NYU":
        config.batch_size = 2
        project_scale = 1
        n_classes = 12
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

    trainer = Trainer(
        sync_batchnorm=True, deterministic=False, devices=config.n_gpus, accelerator="gpu", 
    )

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
    model.eval()
    data_module.setup()
    val_dataloader = data_module.val_dataloader()
    trainer.test(model, dataloaders=val_dataloader)


if __name__ == "__main__":
    main()

import sys
sys.path.append('/home/hongxiao.yu/projects/ISO_occscannet/depth_anything')
from iso.data.NYU.params import (
    class_weights as NYU_class_weights,
    NYU_class_names,
)
from iso.data.OccScanNet.params import (
    class_weights as OccScanNet_class_weights,
    OccScanNet_class_names
)
from iso.data.NYU.nyu_dm import NYUDataModule
from iso.data.OccScanNet.occscannet_dm import OccScanNetDataModule
from torch.utils.data.dataloader import DataLoader
from iso.models.iso import ISO
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
import hydra
from omegaconf import DictConfig
import numpy as np
import torch

hydra.output_subdir = None

pl.seed_everything(658018589)  #, workers=True)

@hydra.main(config_name="../config/iso_occscannet_mini.yaml", config_path='.')
def main(config: DictConfig):
    exp_name = config.exp_prefix
    exp_name += "_{}_{}".format(config.dataset, config.run)
    exp_name += "_FrusSize_{}".format(config.frustum_size)
    exp_name += "_nRelations{}".format(config.n_relations)
    exp_name += "_WD{}_lr{}".format(config.weight_decay, config.lr)

    if config.use_gt_depth:
        exp_name += '_gtdepth'
    if config.add_fusion:
        exp_name += '_addfusion'
    if not config.use_zoedepth:
        exp_name += '_nozoedepth'
    if config.zoedepth_as_gt:
        exp_name += '_zoedepthgt'
    if config.frozen_encoder:
        exp_name += '_frozen_encoder'
    if config.use_depthanything:
        exp_name += '_depthanything'

    voxeldepth_res = []
    if config.voxeldepth:
        exp_name += '_VoxelDepth'
        if config.voxeldepthcfg.depth_scale_1:
            exp_name += '_1'
            voxeldepth_res.append('1')
        if config.voxeldepthcfg.depth_scale_2:
            exp_name += '_2'
            voxeldepth_res.append('2')
        if config.voxeldepthcfg.depth_scale_4:
            exp_name += '_4'
            voxeldepth_res.append('4')
        if config.voxeldepthcfg.depth_scale_8:
            exp_name += '_8'
            voxeldepth_res.append('8')

    if config.CE_ssc_loss:
        exp_name += "_CEssc"
    if config.geo_scal_loss:
        exp_name += "_geoScalLoss"
    if config.sem_scal_loss:
        exp_name += "_semScalLoss"
    if config.fp_loss:
        exp_name += "_fpLoss"

    if config.relation_loss:
        exp_name += "_CERel"
    if config.context_prior:
        exp_name += "_3DCRP"

    # Setup dataloaders
    if config.dataset == "OccScanNet":
        class_names = OccScanNet_class_names
        max_epochs = 60
        logdir = config.logdir
        full_scene_size = (60, 60, 36)
        project_scale = 1
        feature = 200
        n_classes = 12
        class_weights = OccScanNet_class_weights
        data_module = OccScanNetDataModule(
            root=config.OccScanNet_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
        )
    elif config.dataset == "OccScanNet_mini":
        class_names = OccScanNet_class_names
        max_epochs = 60
        logdir = config.logdir
        full_scene_size = (60, 60, 36)
        project_scale = 1
        feature = 200
        n_classes = 12
        class_weights = OccScanNet_class_weights
        data_module = OccScanNetDataModule(
            root=config.OccScanNet_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            train_scenes_sample=4639,
            val_scenes_sample=2007,
        )

    elif config.dataset == "NYU":
        class_names = NYU_class_names
        max_epochs = 30
        logdir = config.logdir
        full_scene_size = (60, 36, 60)
        project_scale = 1
        feature = 200
        n_classes = 12
        class_weights = NYU_class_weights
        data_module = NYUDataModule(
            root=config.NYU_root,
            preprocess_root=config.NYU_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size / config.n_gpus),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
        )

    project_res = ["1"]
    if config.project_1_2:
        exp_name += "_Proj_2"
        project_res.append("2")
    if config.project_1_4:
        exp_name += "_4"
        project_res.append("4")
    if config.project_1_8:
        exp_name += "_8"
        project_res.append("8")

    print(exp_name)
    
    os.chdir(hydra.utils.get_original_cwd())

    # Initialize ISO model
    model = ISO(
        dataset=config.dataset,
        frustum_size=config.frustum_size,
        project_scale=project_scale,
        n_relations=config.n_relations,
        fp_loss=config.fp_loss,
        feature=feature,
        full_scene_size=full_scene_size,
        project_res=project_res,
        voxeldepth=config.voxeldepth,
        voxeldepth_res=voxeldepth_res,
        n_classes=n_classes,
        class_names=class_names,
        context_prior=config.context_prior,
        relation_loss=config.relation_loss,
        CE_ssc_loss=config.CE_ssc_loss,
        sem_scal_loss=config.sem_scal_loss,
        geo_scal_loss=config.geo_scal_loss,
        lr=config.lr,
        weight_decay=config.weight_decay,
        class_weights=class_weights,
        use_gt_depth=config.use_gt_depth,
        add_fusion=config.add_fusion,
        use_zoedepth=config.use_zoedepth,
        use_depthanything=config.use_depthanything,
        zoedepth_as_gt=config.zoedepth_as_gt,
        depthanything_as_gt=config.depthanything_as_gt,
        frozen_encoder=config.frozen_encoder,
    )

    if config.enable_log:
        logger = TensorBoardLogger(save_dir=logdir, name=exp_name, version="")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                monitor="val/mIoU",
                save_top_k=1,
                mode="max",
                filename="{epoch:03d}-{val/mIoU:.5f}",
            ),
            lr_monitor,
        ]
    else:
        logger = False
        checkpoint_callbacks = False

    model_path = os.path.join(logdir, exp_name, "checkpoints/last.ckpt")
    if os.path.isfile(model_path):
        # Continue training from last.ckpt
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            resume_from_checkpoint=model_path,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            devices=config.n_gpus, 
            accelerator="gpu",
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            # flush_logs_every_n_steps=100,
            # strategy="ddp_find_unused_parameters_true",
        )
    else:
        # Train from scratch
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            sync_batchnorm=True,
            deterministic=False,
            max_epochs=max_epochs,
            devices=config.n_gpus,
            accelerator='gpu',
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            # flush_logs_every_n_steps=100,
            strategy="ddp_find_unused_parameters_true",
        )
    torch.set_float32_matmul_precision('high')
    # os.chdir("/home/hongxiao.yu/projects/ISO_occscannet")
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()

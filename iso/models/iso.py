import pytorch_lightning as pl
import torch
import torch.nn as nn
from iso.models.unet3d_nyu import UNet3D as UNet3DNYU
from iso.loss.sscMetrics import SSCMetrics
from iso.loss.ssc_loss import sem_scal_loss, CE_ssc_loss, KL_sep, geo_scal_loss
from iso.models.flosp import FLoSP
from iso.models.depthnet import DepthNet
from iso.loss.CRP_loss import compute_super_CP_multilabel_loss
import numpy as np
import torch.nn.functional as F
from iso.models.unet2d import UNet2D
from torch.optim.lr_scheduler import MultiStepLR
import sys
sys.path.append('./iso')
sys.path.append('./depth_anything/metric_depth')
from depth_anything.metric_depth.zoedepth.models.builder import build_model as build_depthany_model
from depth_anything.metric_depth.zoedepth.utils.config import get_config as get_depthany_config

from iso.models.modules import sample_grid_feature, get_depth_index, sample_3d_feature, bin_depths
# from iso.models.depth_utils import down_sample_depth_dist
from iso.loss.depth_loss import DepthClsLoss

import torch

from torch.cuda.amp import autocast
import torch.nn.functional as F

# from transformers import pipeline
from PIL import Image


class ISO(pl.LightningModule):
    def __init__(
        self,
        n_classes,
        class_names,
        feature,
        class_weights,
        project_scale,
        full_scene_size,
        dataset,
        n_relations=4,
        context_prior=True,
        fp_loss=True,
        project_res=[],
        bevdepth=False,
        voxeldepth=False,
        voxeldepth_res=[],
        frustum_size=4,
        relation_loss=False,
        CE_ssc_loss=True,
        geo_scal_loss=True,
        sem_scal_loss=True,
        lr=1e-4,
        weight_decay=1e-4,
        use_gt_depth=False,
        add_fusion=False,
        use_zoedepth=True,
        use_depthanything=False,
        zoedepth_as_gt=False,
        depthanything_as_gt=False,
        frozen_encoder=False,
    ):
        super().__init__()

        self.project_res = project_res
        self.bevdepth = bevdepth
        self.voxeldepth = voxeldepth
        self.voxeldepth_res = voxeldepth_res
        self.fp_loss = fp_loss
        self.dataset = dataset
        self.context_prior = context_prior
        self.frustum_size = frustum_size
        self.class_names = class_names
        self.relation_loss = relation_loss
        self.CE_ssc_loss = CE_ssc_loss
        self.sem_scal_loss = sem_scal_loss
        self.geo_scal_loss = geo_scal_loss
        self.project_scale = project_scale
        self.class_weights = class_weights
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_gt_depth = use_gt_depth
        self.add_fusion = add_fusion
        self.use_zoedepth = use_zoedepth
        self.use_depthanything = use_depthanything
        self.zoedepth_as_gt = zoedepth_as_gt
        self.depthanything_as_gt = depthanything_as_gt
        self.frozen_encoder = frozen_encoder

        self.projects = {}
        self.scale_2ds = [1, 2, 4, 8]  # 2D scales
        for scale_2d in self.scale_2ds:
            self.projects[str(scale_2d)] = FLoSP(
                full_scene_size, project_scale=self.project_scale, dataset=self.dataset
            )
        self.projects = nn.ModuleDict(self.projects)

        self.n_classes = n_classes
        if self.dataset == "NYU":
            self.net_3d_decoder = UNet3DNYU(
                self.n_classes,
                nn.BatchNorm3d,
                n_relations=n_relations,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            )
            # if self.voxeldepth:
            #     self.depth_net_3d_decoder = UNet3DNYU(
            #         self.n_classes,
            #         nn.BatchNorm3d,
            #         n_relations=n_relations,
            #         feature=feature,
            #         full_scene_size=full_scene_size,
            #         context_prior=context_prior,
            #         beforehead=True,
            #     )
        elif self.dataset == "kitti":
            self.net_3d_decoder = UNet3DKitti(
                self.n_classes,
                nn.BatchNorm3d,
                project_scale=project_scale,
                feature=feature,
                full_scene_size=full_scene_size,
                context_prior=context_prior,
            )
        self.net_rgb = UNet2D.build(out_feature=feature, use_decoder=True, frozen_encoder=self.frozen_encoder)

        if self.voxeldepth and not self.use_gt_depth:
            self.depthnet_1_1 = DepthNet(200, 256, 200, 64)
            if self.use_zoedepth:  # use zoedepth pretrained
                self.net_depth = self._init_zoedepth()
                self.depthnet_1_1 = DepthNet(201, 256, 200, 64)
            elif self.use_depthanything:
                self.net_depth = self._init_depthanything()
                self.depthnet_1_1 = DepthNet(201, 256, 200, 64)
        elif self.zoedepth_as_gt:  # use gt and use zoedepth as gt
            self.net_depth = self._init_zoedepth()
        elif self.depthanything_as_gt:
            self.net_depth = self._init_depthanything()
        elif self.bevdepth:
            self.net_depth = self._init_zoedepth()
        else:
            pass  # use gt and use dataset gt

        # log hyperparameters
        self.save_hyperparameters()

        self.train_metrics = SSCMetrics(self.n_classes)
        self.val_metrics = SSCMetrics(self.n_classes)
        self.test_metrics = SSCMetrics(self.n_classes)
    
    def _init_zoedepth(self):
        conf = get_config("zoedepth", "infer")
        conf['img_size'] = [480, 640]
        model_zoe_n = build_model(conf)
        return model_zoe_n.cuda()
    
    def _init_depthanything(self):
        import sys
        sys.path.append('/home/hongxiao.yu/projects/ISO/depth_anything/metric_depth')
        overrite = {"pretrained_resource": "local::/home/hongxiao.yu/projects/ISO/checkpoints/depth_anything_metric_depth_indoor.pt"}
        conf = get_depthany_config("zoedepth", "infer", "nyu", **overrite)
        # conf['img_size'] = [480, 640]
        from pprint import pprint
        # pprint(conf)
        model = build_depthany_model(conf)
        return model.cuda()
        # return pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
    
    def _set_train_params(self, curr_epoch):
        if curr_epoch // 2 == 1:
            for k, p in self.named_parameters():
                if 'depthnet_1_1' in k:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
        else:
            for k, p in self.named_parameters():
                if 'depthnet_1_1' in k:
                    p.requires_grad = False
                else:
                    p.requires_grad = True

    def forward(self, batch):

        # for k, v in self.state_dict().items():
        #     if v.dtype != torch.float32:
        #         print(k, v.dtype)
        # breakpoint()

        # self._set_train_params(self.current_epoch)

        img = batch["img"]
        raw_img = batch['raw_img']
        pix_z = batch['pix_z']  # (B, 129600)
        bs = len(img)
        # print(img)

        out = {}

        # for k, v in self.named_parameters():
        #     print(k, ':', v)
        x_rgb = self.net_rgb(img)

        x3ds = []
        x3d_bevs = []
        if self.add_fusion:
            x3ds_res = []
        if self.voxeldepth:
            depth_preds = {
                '1_1': [],
                '1_2': [],
                '1_4': [],
                '1_8': [],
            }
        for i in range(bs):
            if self.voxeldepth:
                x3d_depths = {
                    '1_1': None,
                    '1_2': None,
                    '1_4': None,
                    '1_8': None,
                }
                depths = {
                    '1_1': None,
                    '1_2': None,
                    '1_4': None,
                    '1_8': None,
                }
                
                if not self.use_gt_depth:
                    if self.use_zoedepth:
                        # self.net_depth.eval()  # zoe_depth
                        # for param in self.net_depth.parameters():
                        #     param.requires_grad = False
                        # rslts = self.net_depth(raw_img[i:i+1], return_probs=False, return_final_centers=False)
                        # feature = rslts['metric_depth']  # (1, 1, 480, 640)
                        self.net_depth.device = 'cuda'
                        feature = self.net_depth.infer_pil(raw_img[i], output_type="tensor", with_flip_aug=False).cuda().unsqueeze(0).unsqueeze(0)
                        
                        # print(feature.shape)

                        # import matplotlib.pyplot as plt
                        # plt.subplot(1, 2, 1)
                        # plt.imshow(feature[0].permute(1, 2, 0).cpu().numpy())
                        # plt.subplot(1, 2, 2)
                        # plt.imshow(batch['depth_gt'][i].permute(1, 2, 0).cpu().numpy())
                        # plt.savefig('/home/hongxiao.yu/ISO/depth_compare.png')

                        input_kwargs = {
                            "img_feat_1_1": torch.cat([x_rgb['1_1'][i:i+1], feature], dim=1),
                            "cam_k": batch["cam_k"][i:i+1],
                            "T_velo_2_cam": batch["cam_pose"][i:i+1],
                            "vox_origin": batch['vox_origin'][i:i+1],
                        }
                    elif self.use_depthanything:
                        self.net_depth.device = 'cuda'
                        feature = self.net_depth.infer_pil(raw_img[i], output_type="tensor", with_flip_aug=False).cuda().unsqueeze(0).unsqueeze(0)
                        
                        # print(feature.shape)
                        # print(feature.shape)

                        # import matplotlib.pyplot as plt
                        # plt.subplot(1, 2, 1)
                        # plt.imshow(feature[0].permute(1, 2, 0).cpu().numpy())
                        # plt.subplot(1, 2, 2)
                        # plt.imshow(batch['depth_gt'][i].permute(1, 2, 0).cpu().numpy())
                        # plt.savefig('/home/hongxiao.yu/ISO/depth_compare.png')

                        input_kwargs = {
                            "img_feat_1_1": torch.cat([x_rgb['1_1'][i:i+1], feature], dim=1),
                            "cam_k": batch["cam_k"][i:i+1],
                            "T_velo_2_cam": batch["cam_pose"][i:i+1],
                            "vox_origin": batch['vox_origin'][i:i+1],
                        }
                    else:
                        input_kwargs = {
                            "img_feat_1_1": x_rgb['1_1'][i:i+1],
                            "cam_k": batch["cam_k"][i:i+1],
                            "T_velo_2_cam": batch["cam_pose"][i:i+1],
                            "vox_origin": batch['vox_origin'][i:i+1],
                        }
                    intrins_mat = input_kwargs['cam_k'].new_zeros(1, 4, 4).to(torch.float)
                    intrins_mat[:, :3, :3] = input_kwargs['cam_k']
                    intrins_mat[:, 3, 3] = 1  # (1, 4, 4)

                    depth_feature_1_1 = self.depthnet_1_1(
                        x=input_kwargs['img_feat_1_1'],
                        sweep_intrins=intrins_mat,
                        scaled_pixel_size=None,
                    )
                    depths['1_1'] = depth_feature_1_1.softmax(1)  # 得到depth的分布
                    for res in self.voxeldepth_res[1:]:
                        depths['1_'+str(res)] = down_sample_depth_dist(depths['1_1'], int(res))
                else:
                    disc_cfg = {
                        "mode": "LID",
                        "num_bins": 64,
                        "depth_min": 0,
                        "depth_max": 10,
                    }
                    depth_1_1 = batch['depth_gt'][i:i+1]
                    # print(depth_1_1.shape)
                    if self.zoedepth_as_gt:
                        self.net_depth.eval()  # zoe_depth
                        for param in self.net_depth.parameters():
                            param.requires_grad = False
                        self.net_depth.device = 'cuda'
                        rslts = self.net_depth.infer_pil(raw_img[i], output_type="tensor", with_flip_aug=False).cuda()  # TODO: Need Check
                        depth_1_1 = rslts['metric_depth']  # (1, 1, 480, 640)
                    elif self.depthanything_as_gt:
                        self.net_depth.eval()  # zoe_depth
                        for param in self.net_depth.parameters():
                            param.requires_grad = False
                        self.net_depth.device = 'cuda'
                        depth_1_1 = self.net_depth.infer_pil(raw_img[i], output_type="tensor", with_flip_aug=False).cuda().unsqueeze(0).unsqueeze(0)  # TODO: Need Check
                        # depth_1_1 = rslts['metric_depth']  # (1, 1, 480, 640)
                    # print(depth_1_1.shape)
                    depth_1_1 = bin_depths(depth_map=depth_1_1, target=True, **disc_cfg).cuda()  # (1, 1, 480, 640)
                    # depth_1_1 = ((depth_1_1 - (0.1 - 0.13)) / 0.13).cuda().long()
                    # print(depth_1_1.shape)
                    depth_1_1 = F.one_hot(depth_1_1[:, 0, :, :], 81).permute(0, 3, 1, 2)[:, :-1, :, :]  # (1, 81, 480, 640)
                    depths['1_1'] = depth_1_1.float() # 得到depth的分布
                    for res in self.voxeldepth_res[1:]:
                        depths['1_'+str(res)] = down_sample_depth_dist(depths['1_1'], int(res)).float()
                
                projected_pix = batch["projected_pix_{}".format(self.project_scale)][i].cuda()
                fov_mask = batch["fov_mask_{}".format(self.project_scale)][i].cuda()

                # pix_z_index = get_depth_index(pix_z[i])
                disc_cfg = {
                    "mode": "LID",
                    "num_bins": 64,
                    "depth_min": 0,
                    "depth_max": 10,
                }
                pix_z_index = bin_depths(depth_map=pix_z[i], target=True, **disc_cfg).to(fov_mask.device)
                # pix_z_index = ((pix_z[i] - (0.1 - 0.13)) / 0.13).to(fov_mask.device)
                # pix_z_index = torch.where(
                #     (pix_z_index < 80 + 1) & (pix_z_index >= 0.0),
                #     pix_z_index,
                #     torch.zeros_like(pix_z_index),
                # )
                dist_mask = torch.logical_and(pix_z_index >= 0, pix_z_index < 64)
                dist_mask = torch.logical_and(dist_mask, fov_mask)

                for res in self.voxeldepth_res:
                    probs = torch.zeros((129600, 1), dtype=torch.float32).to(self.device)
                    # print(depths['1_'+str(res)].dtype, (projected_pix//int(res)).dtype, pix_z_index.dtype)
                    probs[dist_mask] = sample_3d_feature(depths['1_'+str(res)], projected_pix//int(res), pix_z_index, dist_mask)
                    if self.dataset == 'NYU':
                        x3d_depths['1_'+str(res)] = probs.reshape(60, 60, 36).permute(0, 2, 1).unsqueeze(0)
                    elif self.dataset == 'ScanNet':
                        x3d_depths['1_'+str(res)] = probs.reshape(60, 60, 36).unsqueeze(0)
                    depth_preds['1_'+str(res)].append(depths['1_'+str(res)])  # (1, 64, 60, 80)
            
            x3d = None
            if self.add_fusion:
                x3d_res = None
            for scale_2d in self.project_res:

                # project features at each 2D scale to target 3D scale
                scale_2d = int(scale_2d)
                projected_pix = batch["projected_pix_{}".format(self.project_scale)][i].cuda()
                fov_mask = batch["fov_mask_{}".format(self.project_scale)][i].cuda()
                if self.bevdepth and scale_2d == 4:
                    xys = projected_pix // scale_2d

                    D, fH, fW = 64, 480 // scale_2d, 640 // scale_2d
                    xs = torch.linspace(0, 640 - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW).cuda()  # (64, 120, 160)
                    ys = torch.linspace(0, 480 - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW).cuda()  # (64, 120, 160)
                    d_xs = torch.floor(xs[0].reshape(-1)).to(torch.long)  # (fH*fW,)
                    d_ys = torch.floor(ys[0].reshape(-1)).to(torch.long)  # (fH*fW,)

                    self.net_depth.device = 'cuda'
                    # feature = self.net_depth.infer_pil(raw_img[i], output_type="tensor", with_flip_aug=False).cuda().unsqueeze(0).unsqueeze(0)
                    rslts = self.net_depth(img[i:i+1], return_final_centers=True, return_probs=True)
                    probs = rslts['probs'].cuda()  # 得到depth distribution
                    bin_center = rslts['bin_centers'].cuda()  # (1, 64, 384, 512)
                    # print(probs.shape)
                    probs = probs[0, :, d_ys, d_xs].reshape(D, fH, fW).to(projected_pix.device)  # (D, fH, fW) depth distribution
                    ds = bin_center[0, :, d_ys, d_xs].reshape(D, fH, fW).to(xs.device) # (D, fH, fW) depth bins distance

                    # x = F.interpolate(x_rgb["1_" + str(scale_2d)][i][None, ...], (384, 512), mode='bilinear', align_corners=True)[0]  # (200, 384, 512)
                    x = x_rgb["1_" + str(scale_2d)][i]
                    x = probs.unsqueeze(0) * x.unsqueeze(1)  # (1, 64, 384, 512), (200, 1, 384, 512) --> (200, 64, 384, 512)
                    x = x.reshape(-1, fH, fW)
                    x = self.projects[str(scale_2d)](
                            x,
                            xys,
                            fov_mask,
                        )
                    # print(x.shape)  # (200*64, 60, 60, 36)
                    x = x.reshape(200*64, 60*36*60).T  # (60*60*36, 200*64)
                    pix_z_depth = ds[:, xys[:, 1][fov_mask], xys[:, 0][fov_mask]]  # get the depth bins distance (64, K)
                    pix_zz = pix_z[i][fov_mask]  # (K,)
                    # print(pix_z_depth.shape, pix_z.shape)
                    pix_z_delta = torch.abs(pix_z_depth.to(pix_zz.device) - pix_zz[None, ...])  # (64, K)
                    # print(pix_z_delta.shape)
                    min_z = torch.argmin(pix_z_delta, dim=0)  # (K,)
                    # print(min_z.shape)
                    temp = x[fov_mask].reshape(min_z.shape[0], 200, 64)  # (K, 200, 64)
                    temp = temp[torch.arange(min_z.shape[0]).to(torch.long), :, min_z]  # (K, 200)
                    # temp = temp.reshape(temp.shape[0], 200, 1).repeat(1, 1, 64).reshape(temp.shape[0], 200*64)  # (K, 200*64)
                    # x[fov_mask] = x[fov_mask][torch.arange(min_z.shape[0]), min_z*200:(min_z+1)*200] = temp  # (K, 200*64)
                    # x = x.T.reshape(200*64, 60, 60, 36)[:200, ...]
                    x3d_bev = torch.zeros(60*36*60, 200).to(x.device)
                    x3d_bev[fov_mask] = temp
                    x3d_bev = x3d_bev.T.reshape(200, 60, 36, 60)
                    # torch.cuda.empty_cache()
                
                if self.add_fusion:
                    if x3d_res is None:
                        x3d_res = self.projects[str(scale_2d)](
                            x_rgb["1_" + str(scale_2d)][i],
                            projected_pix // scale_2d,
                            fov_mask,
                        )
                    else:
                        x3d_res += self.projects[str(scale_2d)](
                            x_rgb["1_" + str(scale_2d)][i],
                            projected_pix // scale_2d,
                            fov_mask,
                        )

                # Sum all the 3D features
                if x3d is None:
                    if self.voxeldepth:
                        if len(self.voxeldepth_res) == 1:
                            res = self.voxeldepth_res[0]
                            x3d = self.projects[str(scale_2d)](
                                x_rgb["1_" + str(scale_2d)][i],
                                projected_pix // scale_2d,
                                fov_mask,
                            ) * x3d_depths['1_'+str(res)] * 100
                        else:
                            x3d = self.projects[str(scale_2d)](
                                x_rgb["1_" + str(scale_2d)][i],
                                projected_pix // scale_2d,
                                fov_mask,
                            ) * x3d_depths['1_1'] * 100
                    else:
                        x3d = self.projects[str(scale_2d)](
                            x_rgb["1_" + str(scale_2d)][i],
                            projected_pix // scale_2d,
                            fov_mask,
                        )
                else:
                    if self.voxeldepth:
                        if len(self.voxeldepth_res) == 1:
                            res = self.voxeldepth_res[0]
                            x3d = self.projects[str(scale_2d)](
                                x_rgb["1_" + str(scale_2d)][i],
                                projected_pix // scale_2d,
                                fov_mask,
                            ) * x3d_depths['1_'+str(res)] * 100
                        else:
                            x3d += self.projects[str(scale_2d)](
                                x_rgb["1_" + str(scale_2d)][i],
                                projected_pix // scale_2d,
                                fov_mask,
                            ) * x3d_depths['1_'+str(scale_2d)] * 100
                    else:
                        x3d += self.projects[str(scale_2d)](
                            x_rgb["1_" + str(scale_2d)][i],
                            projected_pix // scale_2d,
                            fov_mask,
                        )
            x3ds.append(x3d)
            if self.add_fusion:
                x3ds_res.append(x3d_res)
            if self.bevdepth:
                x3d_bevs.append(x3d_bev)
        
        input_dict = {
            "x3d": torch.stack(x3ds),
        }
        if self.add_fusion:
            input_dict['x3d'] += torch.stack(x3ds_res)
        if self.bevdepth:
            # print(input_dict['x3d'].shape, torch.stack(x3d_bevs).shape)
            input_dict['x3d'] += torch.stack(x3d_bevs)
        
        # print(input_dict["x3d"])
        # from pytorch_lightning import seed_everything
        # seed_everything(84, True)
        # input_dict['x3d'] = torch.randn_like(input_dict['x3d']).to(torch.float64)
        # print(input_dict['x3d'], 'x3d')

        out = self.net_3d_decoder(input_dict)
        # print(torch.randn((5, 5)), 'x3d')
        # print(torch.randn((5, 5)), 'x3d')

        if self.voxeldepth and not self.use_gt_depth:
            for res in self.voxeldepth_res:
                out['depth_1_'+str(res)] = torch.vstack(depth_preds['1_'+str(res)])  # (B, 64, 60, 80)
            out['depth_gt'] = batch['depth_gt']
        # print(batch['name'], out["ssc_logit"])
        return out

    def step(self, batch, step_type, metric):
        bs = len(batch["img"])
        loss = 0
        out_dict = self(batch)
        ssc_pred = out_dict["ssc_logit"]
        # torch.cuda.manual_seed_all(42)
        # ssc_pred = torch.randn_like(ssc_pred)
        target = batch["target"]

        if self.voxeldepth and not self.use_gt_depth:
            loss_depth_dict = {}
            loss_depth = 0.0
            for res in self.voxeldepth_res:
                loss_depth_dict['1_'+str(res)] = DepthClsLoss(int(res), [0.0, 10, 0.16], 64).get_depth_loss(out_dict['depth_gt'], 
                                                                                                 out_dict['depth_1_'+str(res)].unsqueeze(1))
                loss_depth += loss_depth_dict['1_'+str(res)]
            loss_depth = loss_depth / len(loss_depth_dict)
            loss += loss_depth
            self.log(
                step_type + "/loss_depth",
                loss_depth.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.context_prior:
            P_logits = out_dict["P_logits"]
            CP_mega_matrices = batch["CP_mega_matrices"]

            if self.relation_loss:
                loss_rel_ce = compute_super_CP_multilabel_loss(
                    P_logits, CP_mega_matrices
                )
                loss += loss_rel_ce
                self.log(
                    step_type + "/loss_relation_ce_super",
                    loss_rel_ce.detach(),
                    on_epoch=True,
                    sync_dist=True,
                )

        class_weight = self.class_weights.type_as(batch["img"])
        if self.CE_ssc_loss:
            loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
            loss += loss_ssc
            self.log(
                step_type + "/loss_ssc",
                loss_ssc.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.sem_scal_loss:
            loss_sem_scal = sem_scal_loss(ssc_pred, target)
            loss += loss_sem_scal
            self.log(
                step_type + "/loss_sem_scal",
                loss_sem_scal.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.geo_scal_loss:
            loss_geo_scal = geo_scal_loss(ssc_pred, target)
            loss += loss_geo_scal
            self.log(
                step_type + "/loss_geo_scal",
                loss_geo_scal.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.fp_loss and step_type != "test":
            frustums_masks = torch.stack(batch["frustums_masks"])
            frustums_class_dists = torch.stack(
                batch["frustums_class_dists"]
            ).float()  # (bs, n_frustums, n_classes)
            n_frustums = frustums_class_dists.shape[1]

            pred_prob = F.softmax(ssc_pred, dim=1)
            batch_cnt = frustums_class_dists.sum(0)  # (n_frustums, n_classes)

            frustum_loss = 0
            frustum_nonempty = 0
            for frus in range(n_frustums):
                frustum_mask = frustums_masks[:, frus, :, :, :].unsqueeze(1).float()
                prob = frustum_mask * pred_prob  # bs, n_classes, H, W, D
                prob = prob.reshape(bs, self.n_classes, -1).permute(1, 0, 2)
                prob = prob.reshape(self.n_classes, -1)
                cum_prob = prob.sum(dim=1)  # n_classes

                total_cnt = torch.sum(batch_cnt[frus])
                total_prob = prob.sum()
                if total_prob > 0 and total_cnt > 0:
                    frustum_target_proportion = batch_cnt[frus] / total_cnt
                    cum_prob = cum_prob / total_prob  # n_classes
                    frustum_loss_i = KL_sep(cum_prob, frustum_target_proportion)
                    frustum_loss += frustum_loss_i
                    frustum_nonempty += 1
            frustum_loss = frustum_loss / frustum_nonempty
            loss += frustum_loss
            self.log(
                step_type + "/loss_frustums",
                frustum_loss.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        y_true = target.cpu().numpy()
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        metric.add_batch(y_pred, y_true)

        self.log(step_type + "/loss", loss.detach(), on_epoch=True, sync_dist=True, prog_bar=True)
        
        # loss_dict = {'loss': loss}

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metrics)

    def on_validation_epoch_end(self):  #, outputs):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]

        for prefix, metric in metric_list:
            stats = metric.get_stats()
            for i, class_name in enumerate(self.class_names):
                self.log(
                    "{}_SemIoU/{}".format(prefix, class_name),
                    stats["iou_ssc"][i],
                    sync_dist=True,
                )
            self.log("{}/mIoU".format(prefix), torch.tensor(stats["iou_ssc_mean"]).to(torch.float32), sync_dist=True)
            self.log("{}/IoU".format(prefix), torch.tensor(stats["iou"]).to(torch.float32), sync_dist=True)
            self.log("{}/Precision".format(prefix), torch.tensor(stats["precision"]).to(torch.float32), sync_dist=True)
            self.log("{}/Recall".format(prefix), torch.tensor(stats["recall"]).to(torch.float32), sync_dist=True)
            metric.reset()

    def test_step(self, batch, batch_idx):
        self.step(batch, "test", self.test_metrics)

    def on_test_epoch_end(self,):# outputs):
        classes = self.class_names
        metric_list = [("test", self.test_metrics)]
        for prefix, metric in metric_list:
            print("{}======".format(prefix))
            stats = metric.get_stats()
            print(
                "Precision={:.4f}, Recall={:.4f}, IoU={:.4f}".format(
                    stats["precision"] * 100, stats["recall"] * 100, stats["iou"] * 100
                )
            )
            print("class IoU: {}, ".format(classes))
            print(
                " ".join(["{:.4f}, "] * len(classes)).format(
                    *(stats["iou_ssc"] * 100).tolist()
                )
            )
            print("mIoU={:.4f}".format(stats["iou_ssc_mean"] * 100))
            metric.reset()

    def configure_optimizers(self):
        if self.dataset == "NYU" and not self.voxeldepth:
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
            return [optimizer], [scheduler]
        elif self.dataset == "NYU" and self.voxeldepth:
            depth_params = []
            other_params = []
            for k, p in self.named_parameters():
                if 'depthnet_1_1' in k:
                    depth_params.append(p)
                else:
                    other_params.append(p)
            params_list = [{'params': depth_params, 'lr': self.lr * 0.05},  # 0.4 high, 0.1 unstable, 0.05 ok
                    {'params': other_params}]
            optimizer = torch.optim.AdamW(
                params_list, lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
            return [optimizer], [scheduler]
        elif self.dataset == "kitti":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
            return [optimizer], [scheduler]

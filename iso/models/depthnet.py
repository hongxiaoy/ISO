import torch
import torch.nn as nn
import torch.nn.functional as F
# from mmdet.models.backbones.resnet import BasicBlock

import math
# from occdepth.models.f2v.frustum_grid_generator import FrustumGridGenerator
# from occdepth.models.f2v.frustum_to_voxel import FrustumToVoxel
# from occdepth.models.f2v.sampler import Sampler

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                #  style='pytorch',
                 with_cp=False,
                #  conv_cfg=None,
                #  norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None,
                 init_cfg=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        """nn.Module: normalization layer after the first convolution layer"""
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        """nn.Module: normalization layer after the second convolution layer"""
        return getattr(self, self.norm2_name)

    def forward(self, x):
        """Forward function."""
        # print(x.shape)

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        out = _inner_forward(x)

        out = self.relu(out)

        return out


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        context_channels,
        depth_channels,
        infer_mode=False,
    ):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.mlp = Mlp(1, mid_channels, mid_channels)
        self.se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        )
        # self.aspp = ASPP(mid_channels, mid_channels, BatchNorm=nn.InstanceNorm2d)

        self.depth_pred = nn.Conv2d(
            mid_channels, depth_channels, kernel_size=1, stride=1, padding=0
        )
        self.infer_mode = infer_mode

    def forward(
        self,
        x=None,
        sweep_intrins=None,
        scaled_pixel_size=None,
        scale_depth_factor=1000.0,
    ):
        # self.eval()
        inv_intrinsics = torch.inverse(sweep_intrins)
        pixel_size = torch.norm(
            torch.stack(
                [inv_intrinsics[..., 0, 0], inv_intrinsics[..., 1, 1]], dim=-1
            ),
            dim=-1,
        ).reshape(-1, 1).to(x.device)
        scaled_pixel_size = pixel_size * scale_depth_factor

        x = self.reduce_conv(x)
        # aug_scale = torch.sqrt(sweep_post_rots_ida[..., 0, 0] ** 2 + sweep_post_rots_ida[..., 0, 1] ** 2).reshape(-1, 1)
        x_se = self.mlp(scaled_pixel_size)[..., None, None]

        x = self.se(x, x_se)
        x = self.depth_conv(x)
        # x = self.aspp(x)
        depth = self.depth_pred(x)
        return depth

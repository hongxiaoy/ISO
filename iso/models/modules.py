import torch
import torch.nn as nn
import math
from monoscene.models.DDR import Bottleneck3D


class ASPP(nn.Module):
    """
    ASPP 3D
    Adapt from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/models/LMSCNet.py#L7
    """

    def __init__(self, planes, dilations_conv_list):
        super().__init__()

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.relu = nn.ReLU()

    def forward(self, x_in):

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        return x_in


class SegmentationHead(nn.Module):
    """
    3D Segmentation heads to retrieve semantic segmentation at each scale.
    Formed by Dim expansion, Conv3D, ASPP block, Conv3D.
    Taken from https://github.com/cv-rits/LMSCNet/blob/main/LMSCNet/models/LMSCNet.py#L7
    """

    def __init__(self, inplanes, planes, nbr_classes, dilations_conv_list):
        super().__init__()

        # First convolution
        self.conv0 = nn.Conv3d(inplanes, planes, kernel_size=3, padding=1, stride=1)

        # ASPP Block
        self.conv_list = dilations_conv_list
        self.conv1 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn1 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.conv2 = nn.ModuleList(
            [
                nn.Conv3d(
                    planes, planes, kernel_size=3, padding=dil, dilation=dil, bias=False
                )
                for dil in dilations_conv_list
            ]
        )
        self.bn2 = nn.ModuleList(
            [nn.BatchNorm3d(planes) for dil in dilations_conv_list]
        )
        self.relu = nn.ReLU()

        self.conv_classes = nn.Conv3d(
            planes, nbr_classes, kernel_size=3, padding=1, stride=1
        )

    def forward(self, x_in):

        # Convolution to go from inplanes to planes features...
        x_in = self.relu(self.conv0(x_in))

        y = self.bn2[0](self.conv2[0](self.relu(self.bn1[0](self.conv1[0](x_in)))))
        for i in range(1, len(self.conv_list)):
            y += self.bn2[i](self.conv2[i](self.relu(self.bn1[i](self.conv1[i](x_in)))))
        x_in = self.relu(y + x_in)  # modified

        x_in = self.conv_classes(x_in)

        return x_in


class Process(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, dilations=[1, 2, 3]):
        super(Process, self).__init__()
        self.main = nn.Sequential(
            *[
                Bottleneck3D(
                    feature,
                    feature // 4,
                    bn_momentum=bn_momentum,
                    norm_layer=norm_layer,
                    dilation=[i, i, i],
                )
                for i in dilations
            ]
        )

    def forward(self, x):
        return self.main(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, bn_momentum):
        super(Upsample, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=1,
            ),
            norm_layer(out_channels, momentum=bn_momentum),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main(x)


class Downsample(nn.Module):
    def __init__(self, feature, norm_layer, bn_momentum, expansion=8):
        super(Downsample, self).__init__()
        self.main = Bottleneck3D(
            feature,
            feature // 4,
            bn_momentum=bn_momentum,
            expansion=expansion,
            stride=2,
            downsample=nn.Sequential(
                nn.AvgPool3d(kernel_size=2, stride=2),
                nn.Conv3d(
                    feature,
                    int(feature * expansion / 4),
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                norm_layer(int(feature * expansion / 4), momentum=bn_momentum),
            ),
            norm_layer=norm_layer,
        )

    def forward(self, x):
        return self.main(x)


def sample_grid_feature(feature, fho=480, fwo=640, scale=4):
    """
    Args:
        feature (torch.tensor): 2D feature to be sampled, shape (B, C, H, W)
    """
    if len(feature.shape) == 4:
        B, D, H, W = feature.shape
    else:
        D, H, W = feature.shape
    fH, fW = fho // scale, fwo // scale
    xs = torch.linspace(0, fwo - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  # (64, 120, 160)
    ys = torch.linspace(0, fho - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)  # (64, 120, 160)
    d_xs = torch.floor(xs[0].reshape(-1)).to(torch.long)  # (fH*fW,)
    d_ys = torch.floor(ys[0].reshape(-1)).to(torch.long)  # (fH*fW,)
    # grid_pts = torch.stack([d_xs, d_ys], dim=1)  # (fH*fW, 2)

    if len(feature.shape) == 4:
        sample_feature = feature[:, :, d_ys, d_xs].reshape(B, D, fH, fW)  # (D, fH, fW)
    else:
        sample_feature = feature[:, d_ys, d_xs].reshape(D, fH, fW)
    return sample_feature


def get_depth_index(pix_z):
    """
    Args:
        pix_z (torch.tensor): The depth in camera frame after voxel projected to pixel, shape (N,), N is 
            total voxel number.
    """
    ds = torch.arange(64).to(pix_z.device)  # (64,)
    ds = 10 / 64 / 65 * ds * (ds + 1)  # (64,)
    delta_z = torch.abs(pix_z[None, ...] - ds[..., None])  # (64, N)
    pix_z_index = torch.argmin(delta_z, dim=0)  # (N,)
    return pix_z_index


def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = (depth_map - depth_min) / bin_size
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = (
            num_bins
            * (torch.log(1 + depth_map) - math.log(1 + depth_min))
            / (math.log(1 + depth_max) - math.log(1 + depth_min))
        )
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds (-2, -1, 0, 1, ..., num_bins, num_bins +1) --> (num_bins, num_bins, 0, 1, ..., num_bins, num_bins)
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = indices.type(torch.int64)
    return indices.long()


def sample_3d_feature(feature_3d, pix_xy, pix_z, fov_mask):
    """
    Args:
        feature_3d (torch.tensor): 3D feature, shape (C, D, H, W).
        pix_xy (torch.tensor): Projected pix coordinate, shape (N, 2).
        pix_z (torch.tensor): Projected pix depth coordinate, shape (N,).
    
    Returns:
        torch.tensor: Sampled feature, shape (N, C)
    """
    pix_x, pix_y = pix_xy[:, 0][fov_mask], pix_xy[:, 1][fov_mask]
    pix_z = pix_z[fov_mask].to(pix_y.dtype)
    ret = feature_3d[:, pix_z, pix_y, pix_x].T
    return ret


import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F

from iso.models.modules import bin_depths


class DepthClsLoss:
    def __init__(self, downsample_factor, d_bound, depth_channels=64):
        self.downsample_factor = downsample_factor
        self.depth_channels = depth_channels
        self.d_bound = d_bound
        # self.depth_channels = int((self.d_bound[1] - self.d_bound[0]) / self.d_bound[2])

    def _get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // self.downsample_factor,
            self.downsample_factor,
            W // self.downsample_factor,
            self.downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample_factor * self.downsample_factor)

        gt_depths_tmp = torch.where(
            gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths
        )
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(
            B * N, H // self.downsample_factor, W // self.downsample_factor
        )

        # ds = torch.arange(64)  # (64,)
        # depth_bin_pos = (10.0 / 64 / 65 * ds * (ds + 1)).reshape(1, 64, 1, 1)
        # # print(gt_depths.unsqueeze(1).shape)
        # # print(depth_bin_pos.shape)
        # delta_z = torch.abs(gt_depths.unsqueeze(1) - depth_bin_pos.to(gt_depths.device))
        # gt_depths = torch.argmin(delta_z, dim=1)
        gt_depths = bin_depths(gt_depths, mode='LID', depth_min=0, depth_max=10, num_bins=self.depth_channels, target=True)

        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths,
            torch.zeros_like(gt_depths),
        )
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.depth_channels + 1
        ).view(-1, self.depth_channels + 1)[:, :-1]

        return gt_depths.float()

    def get_depth_loss(self, depth_labels, depth_preds):
        if len(depth_labels.shape) != 4:
            depth_labels = depth_labels.unsqueeze(1)
            # print(depth_labels.shape)
        N_pred, n_cam_pred, D, H, W = depth_preds.shape
        N_gt, n_cam_label, oriH, oriW = depth_labels.shape
        assert (
            N_pred * n_cam_pred == N_gt * n_cam_label
        ), f"N_pred: {N_pred}, n_cam_pred: {n_cam_pred}, N_gt: {N_gt}, n_cam_label: {n_cam_label}"
        depth_labels = depth_labels.reshape(N_gt * n_cam_label, oriH, oriW)
        depth_preds = depth_preds.reshape(N_pred * n_cam_pred, D, H, W)

        # depth_labels = depth_labels.reshape(
        #     N
        # )

        # depth_labels = depth_labels.unsqueeze(1)
        # depth_labels = depth_labels
        depth_labels = F.interpolate(
            depth_labels.unsqueeze(1),
            (H * self.downsample_factor, W * self.downsample_factor),
            mode="nearest",
        )
        depth_labels = self._get_downsampled_gt_depth(depth_labels)
        depth_preds = (
            depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_channels)
        )
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds[fg_mask],
                depth_labels[fg_mask],
                reduction="none",
            ).sum() / max(1.0, fg_mask.sum())
            # depth_loss = torch.nan_to_num(depth_loss, nan=0.)

        return depth_loss

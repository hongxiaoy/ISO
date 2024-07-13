import pickle
import os
from omegaconf import DictConfig
import numpy as np
import hydra
# from mayavi import mlab
import open3d as o3d


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """
    
    # The sensor in centered in X (we go to dims/2 + 1 for the histogramdd)
    g_xx = np.arange(0, dims[0] + 1)
    # The sensor is in Y=0 (we go to dims + 1 for the histogramdd)
    g_yy = np.arange(0, dims[1] + 1)
    # The sensor is in Z=1.73. I observed that the ground was to voxel levels above the grid bottom, so Z pose is at 10
    # if bottom voxel is 0. If we want the sensor to be at (0, 0, 0), then the bottom in z is -10, top is 22
    # (we go to 22 + 1 for the histogramdd)
    # ATTENTION.. Is 11 for old grids.. 10 for new grids (v1.1) (https://github.com/PRBonn/semantic-kitti-api/issues/49)
    g_zz = np.arange(0, dims[2] + 1)

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float)

    coords_grid = (coords_grid * resolution) + resolution / 2

    temp = np.copy(coords_grid)
    temp[:, 0] = coords_grid[:, 1]
    temp[:, 1] = coords_grid[:, 0]
    coords_grid = np.copy(temp)

    return coords_grid



# def draw_semantic_open3d(
#     voxels,
#     cam_param_path="",
#     voxel_size=0.2):    



#     grid_coords, _, _, _ = get_grid_coords([voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size)    

#     points = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

#     # Obtaining voxels with semantic class
#     points = points[(points[:, 3] != 0) & (points[:, 3] != 255)] # remove empty voxel and unknown class

#     colors = np.take_along_axis(colors, points[:, 3].astype(np.int32).reshape(-1, 1), axis=0)
    
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(width=1200, height=600)
#     ctr = vis.get_view_control()
#     param = o3d.io.read_pinhole_camera_parameters(cam_param_path)

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points[:, :3])
#     pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
#     pcd.estimate_normals()
#     vis.add_geometry(pcd)

#     ctr.convert_from_pinhole_camera_parameters(param)

#     vis.run()  # user changes the view and press "q" to terminate
#     param = vis.get_view_control().convert_to_pinhole_camera_parameters()
#     o3d.io.write_pinhole_camera_parameters(cam_param_path, param)





def draw(
    voxels,
    cam_pose,
    vox_origin,
    voxel_size=0.08,
    d=0.75,  # 0.75m - determine the size of the mesh representing the camera
):
    # Compute the coordinates of the mesh representing camera
    y = d * 480 / (2 * 518.8579)
    x = d * 640 / (2 * 518.8579)
    tri_points = np.array(
        [
            [0, 0, 0],
            [x, y, d],
            [-x, y, d],
            [-x, -y, d],
            [x, -y, d],
        ]
    )
    tri_points = np.hstack([tri_points, np.ones((5, 1))])

    tri_points = (cam_pose @ tri_points.T).T
    x = tri_points[:, 0] - vox_origin[0]
    y = tri_points[:, 1] - vox_origin[1]
    z = tri_points[:, 2] - vox_origin[2]
    triangles = [
        (0, 1, 2),
        (0, 1, 4),
        (0, 3, 4),
        (0, 2, 3),
    ]
    

    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[2], voxels.shape[1]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack(
        (grid_coords.T, np.moveaxis(voxels, [0, 1, 2], [0, 2, 1]).reshape(-1))
    ).T

    # Remove empty and unknown voxels
    occupied_voxels = grid_coords[(grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)]

#     # Draw the camera
#     mlab.triangular_mesh(
#         x,
#         y,
#         z,
#         triangles,
#         representation="wireframe",
#         color=(0, 0, 0),
#         line_width=5,
#     )

#     # Draw occupied voxels
#     plt_plot = mlab.points3d(
#         occupied_voxels[:, 0],
#         occupied_voxels[:, 1],
#         occupied_voxels[:, 2],
#         occupied_voxels[:, 3],
#         colormap="viridis",
#         scale_factor=voxel_size - 0.1 * voxel_size,
#         mode="cube",
#         opacity=1.0,
#         vmin=0,
#         vmax=12,
#     )

    colors = np.array(
        [
            [22, 191, 206, 255],
            [214, 38, 40, 255],
            [43, 160, 43, 255],
            [158, 216, 229, 255],
            [114, 158, 206, 255],
            [204, 204, 91, 255],
            [255, 186, 119, 255],
            [147, 102, 188, 255],
            [30, 119, 181, 255],
            [188, 188, 33, 255],
            [255, 127, 12, 255],
            [196, 175, 214, 255],
            [153, 153, 153, 255],
        ]
    ) / 255.0
    
    colors = np.take_along_axis(colors, occupied_voxels[:, 3].astype(np.int32).reshape(-1, 1), axis=0)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=600)
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(cam_pose)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(occupied_voxels[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    pcd.estimate_normals()
    vis.add_geometry(pcd)

    ctr.convert_from_pinhole_camera_parameters(cam_pose)

    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters(cam_param_path, param)


@hydra.main(config_path=None)
def main(config: DictConfig):
    scan = config.file

    with open(scan, "rb") as handle:
        b = pickle.load(handle)

    cam_pose = b["cam_pose"]
    vox_origin = b["vox_origin"]
    gt_scene = b["target"]
    pred_scene = b["y_pred"]
    scan = os.path.basename(scan)[:12]

    pred_scene[(gt_scene == 255)] = 255  # only draw scene inside the room

    draw(
        pred_scene,
        cam_pose,
        vox_origin,
        voxel_size=0.08,
        d=0.75,
    )


if __name__ == "__main__":
    main()

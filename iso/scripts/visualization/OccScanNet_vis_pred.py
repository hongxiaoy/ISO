import numpy as np
from mayavi import mlab
import argparse


def load_voxels(path):
    """Load voxel labels from file.

    Args:
        path (str): The path of the voxel labels file.
    
    Returns:
        ndarray: The voxel labels with shape (N, 4), 4 is for [x, y, z, label].
    """
    labels = np.load(path)
    if labels.shape[1] == 7:
        labels = labels[:, [0, 1, 2, 6]]
    
    return labels


def draw(voxel_label, voxel_size=0.05, intrinsic=None, cam_pose=None, d=0.5):
    """Visualize the gt or predicted voxel labels.
    
    Args:
        voxel_label (ndarray): The gt or predicted voxel label, with shape (N, 4), N is for number 
            of voxels, 7 is for [x, y, z, label].
        voxel_size (double): The size of each voxel.
        intrinsic (ndarray): The camera intrinsics.
        cam_pose (ndarray): The camera pose.
        d (double): The depth of camera model visualization.
    """
    figure = mlab.figure(size=(1600*0.8, 900*0.8), bgcolor=(1, 1, 1))
    
    # voxel_origin = np.array([-0.6619388,
    #                         -2.3863946,
    #                         -0.05     ])
    # voxel_1 = voxel_origin + np.array([4.8, 0, 0])
    # voxel_2 = voxel_origin + np.array([0, 4.8, 0])
    # voxel_3 = voxel_origin + np.array([0, 0, 4.8])
    # voxel_4 = voxel_origin + np.array([4.8, 4.8, 0])
    # voxel_5 = voxel_origin + np.array([4.8, 0, 4.8])
    # voxel_6 = voxel_origin + np.array([0, 4.8, 4.8])
    # voxel_7 = voxel_origin + np.array([4.8, 4.8, 4.8])
    # voxels = np.vstack([voxel_origin, voxel_1, voxel_2, voxel_3, voxel_4, voxel_5, voxel_6, voxel_7])
    # print(voxels.shape)
    # x = voxels[:, 0]
    # y = voxels[:, 1]
    # z = voxels[:, 2]
    # sqs = [
    #     (0, 1, 2),
    #     (0, 1, 3),
    #     (0, 2, 3),
    #     (1, 2, 4),
    #     (1, 3, 5),
    #     (2, 3, 6),
    #     (3, 5, 6),
    #     (5, 6, 7),
    # ]

    # # draw cam model
    # mlab.triangular_mesh(
    #     x,
    #     y,
    #     z,
    #     sqs,
    #     representation="wireframe",
    #     color=(1, 0, 0),
    #     line_width=7.5,
    # )
    
    
    if intrinsic is not None and cam_pose is not None:
        assert d > 0, 'camera model d should > 0'
        fx = intrinsic[0, 0]
        fy = intrinsic[1, 1]
        cx = intrinsic[0, 2]
        cy = intrinsic[1, 2]

        # half of the image plane size
        y = d * 2 * cy / (2 * fy)
        x = d * 2 * cx / (2 * fx)
        
        # camera points (cam frame)
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
        
        # camera points (world frame)
        tri_points = (cam_pose @ tri_points.T).T
        x = tri_points[:, 0]
        y = tri_points[:, 1]
        z = tri_points[:, 2]
        triangles = [
            (0, 1, 2),
            (0, 1, 4),
            (0, 3, 4),
            (0, 2, 3),
        ]

        # draw cam model
        mlab.triangular_mesh(
            x,
            y,
            z,
            triangles,
            representation="wireframe",
            color=(0, 0, 0),
            line_width=7.5,
        )
    
    # draw occupied voxels
    plt_plot = mlab.points3d(
        voxel_label[:, 0],
        voxel_label[:, 1],
        voxel_label[:, 2],
        voxel_label[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.1 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=12,
    )

    # label colors
    colors = np.array(
        [
            [0, 0, 0, 255],  # 0 empty
            [255, 202, 251, 255],  # 1 ceiling
            [208, 192, 122, 255],  # 2 floor
            [199, 210, 255, 255],  # 3 wall
            [82, 42, 127, 255],  # 4 window
            [224, 250, 30, 255],  # 5 chair
            [255, 0, 65, 255],  # 6  bed
            [144, 177, 144, 255],  # 7 sofa
            [246, 110, 31, 255],  # 8 table
            [0, 216, 0, 255],  # 9 tv
            [135, 177, 214, 255],  # 10 furniture
            [1, 92, 121, 255],  # 11 objects
            [128, 128, 128, 255],  # 12 occupied with semantic
        ]
    )

    plt_plot.glyph.scale_mode = "scale_by_vector"

    plt_plot.module_manager.scalar_lut_manager.lut.table = colors

    mlab.show()


def parse_args():
    parser = argparse.ArgumentParser(description="CompleteScanNet dataset visualization.")
    parser.add_argument("--file", type=str, help="Voxel label file path.", required=True)
    args = parser.parse_args()  
    return args


if __name__ == "__main__":
    args = parse_args()
    voxels = load_voxels(args.file)
    draw(voxels, voxel_size=0.05, d=0.5)
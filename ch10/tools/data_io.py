import os
import numpy as np
from ch10.tools.rotation import rotate_point_se3_by_angle_axis


def read_custombudle_from_txt():
    txt_file = os.path.join(os.path.dirname(__file__),
                            '../data/problem-16-22106-pre.txt')

    with open(txt_file, 'r') as fp:
        lines = list(fp.readlines())

        n_cams, n_pts_3d, n_obs = [int(x) for x in lines[0].strip().split()]

        # 1. read 3D points to 2D pixels projections
        cam_proj_obs = {}
        for i in range(1, n_obs + 1):
            cam_id, pt_3d_id, pt_2d_u, pt_2d_v = lines[i].strip().split()

            cam_id = int(cam_id)
            pt_3d_id = int(pt_3d_id)
            pt_2d_u = float(eval(pt_2d_u))
            pt_2d_v = float(eval(pt_2d_v))

            if cam_id not in cam_proj_obs:
                cam_proj_obs[cam_id] = {
                    'point_3d_id': [],
                    'point_2d_pixel': []
                }
            cam_proj_obs[cam_id]['point_3d_id'].append(pt_3d_id)
            cam_proj_obs[cam_id]['point_2d_pixel'].append([pt_2d_u, pt_2d_v])

        # 2. read camera 3D poses
        i = n_obs + 1
        cam_se3_params = np.array(
            [float(eval(x.strip())) for x in lines[i:i + 9 * n_cams]],
            dtype=np.float64).reshape([-1, 9])

        # 3. read 3D point coordinates
        i = n_obs + 1 + 9 * n_cams
        pts_3d = np.array(
            [float(eval(x.strip())) for x in lines[i:i + 3 * n_pts_3d]],
            dtype=np.float64).reshape([-1, 3])
        return cam_proj_obs, cam_se3_params, pts_3d


def write_as_ply_file(vtxs, ply_file):
    dir_name = os.path.dirname(ply_file)
    if len(dir_name) > 0:
        os.makedirs(dir_name, exist_ok=True)

    with open(ply_file, 'w', encoding='ascii') as fp:
        headers = [
            'ply', \
            'format ascii 1.0', \
            'element face 0', \
            'property list uchar int vertex_indices', \
            'element vertex %d' % len(vtxs), \
            'property float x', \
            'property float y', \
            'property float z', \
            'property uchar red', \
            'property uchar green',
            'property uchar blue', \
            'property uchar alpha', \
            'end_header\n'
        ]

        fp.write('\n'.join(headers))

        for x, y, z, r, g, b, a in vtxs:
            fp.write('%f %f %f %d %d %d %d\n' % (x, y, z, r, g, b, a))


def write_custombudle_to_ply(cam_se3_params, pts_3d, ply_file):
    vtxs = []

    # calculate camera origin coordinate in world frame
    # given 3D transform (R, t) from world to camera frame
    #     R @ X + t = 0 -> X = -R^T @ t
    for cam_param in cam_se3_params:
        rot_vec_w = cam_param[:3]
        t_vec_w = cam_param[3:6]

        cam_cx_w, cam_cy_w, cam_cz_w = \
            -1.0 * rotate_point_se3_by_angle_axis(rot_vec_w, t_vec_w)

        vtxs.append([cam_cx_w, cam_cy_w, cam_cz_w, 0, 255, 0, 255])

    for pt_3d_x, pt_3d_y, pt_3d_z in pts_3d:
        vtxs.append([pt_3d_x, pt_3d_y, pt_3d_z, 255, 255, 255, 255])

    write_as_ply_file(vtxs, ply_file)
import os
import cv2 as cv
import numpy as np


def rotate_point_se3_by_angle_axis(rot_vec, pt_3d_old, epsilon=1e-6):
    theta = np.linalg.norm(rot_vec)

    if theta > epsilon:
        # case-1: far away from zero, use SO(3) rodrigues formula
        # given so(3) lie-algebra:
        #     φ = θ * vec(a)
        # corresponding SO(3) lie-group exponential mapping:
        #     J = cosθ * I + (1 - cosθ) * a @ a.T + sinθ * a^
        rot_mat, _ = cv.Rodrigues(rot_vec)

        pt_3d_new = np.squeeze(rot_mat @ pt_3d_old[:, None])
    else:
        # case-2: near zero, use taylor expansion 1st-order approximation
        # corresponding SO(3) lie-group exponential mapping:
        #     J ~= I + sinθ * a^ ~= I + θ * a^
        pt_3d_new = pt_3d_old + np.cross(rot_vec, pt_3d_old)
    return pt_3d_new
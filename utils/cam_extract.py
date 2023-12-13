import math

import cam_utils as utils
import cv2
import numpy as np


def compute_E_from_F(F_mat, K1, K2):
    E_mat = K2.T @ F_mat @ K1
    return E_mat


def compute_rt(K1_mat, K2_mat, F_mat, correspondences):
    E_mat = compute_E_from_F(F_mat, K1=K1_mat, K2=K2_mat)

    raw_pts1, raw_pts2 = correspondences[:, :2], correspondences[:, 2:]
    pts1 = raw_pts1[:, ::-1]  # front
    pts2 = raw_pts2[:, ::-1]  # back

    # correspondences is [N x 4]
    points, R_mat, t_mat, mask = cv2.recoverPose(
        E_mat, np.float32(pts2), np.float32(pts1), K1_mat, 50
    )
    return R_mat, t_mat


def r_to_euler(pose_mat):
    cosine_for_pitch = math.sqrt(pose_mat[0][0] ** 2 + pose_mat[1][0] ** 2)
    is_singular = cosine_for_pitch < 10**-6
    if not is_singular:
        yaw = math.atan2(pose_mat[1][0], pose_mat[0][0])
        pitch = math.atan2(-pose_mat[2][0], cosine_for_pitch)
        roll = math.atan2(pose_mat[2][1], pose_mat[2][2])
    else:
        yaw = math.atan2(-pose_mat[1][2], pose_mat[1][1])
        pitch = math.atan2(-pose_mat[2][0], cosine_for_pitch)
        roll = 0
    # xyz_angles = [np.rad2deg(pitch), np.rad2deg(roll), np.rad2deg(yaw)]
    return [np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw)]


if __name__ == "__main__":
    """
    [[ 0.51128958 -0.01874729 -0.859204  ]
     [-0.85933165  0.00221703 -0.51141392]
     [ 0.01149251  0.9998218  -0.01497659]]

     [[-0.51138646]
     [ 0.85934243]
     [ 0.00380577]]
    """

    correspondences = utils.read_data(
        r"D:\adev\virtual-correspondences\images\messi\correspondences.npy"
    )
    F_mat = utils.read_data(
        r"D:\adev\virtual-correspondences\images\out\messi_f_mat.npy"
    )

    K1_mat = np.array(
        [
            [1.0311, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )

    K2_mat = np.array(
        [
            [1, 0, 0],
            [0, 1.2522, 0],
            [0, 0, 1],
        ]
    )

    R_mat, t_mat = compute_rt(K1_mat, K2_mat, F_mat, correspondences)
    print(R_mat, t_mat)
    print(r_to_euler(R_mat))

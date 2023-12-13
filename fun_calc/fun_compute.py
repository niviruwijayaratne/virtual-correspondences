import os
import sys

import cv2
import fun_utils as utils
import numpy as np
from tqdm import tqdm

NUM_ITERS = 50000
ERROR_TOL = 0.3  # messi 7 (rad 10), kobe 20 (rad 20), gwh 6 (rad 20)
IN_DIR = r"D:\adev\virtual-correspondences\images"
OUT_DIR = r"D:\adev\virtual-correspondences\images\out"


def _line_point_distances(lines, pts):
    # implementing the distance of a point to a line formula
    dist_num = np.abs(np.sum(lines * pts, axis=1))
    dist_denom = np.linalg.norm(lines[:, :-1], axis=1)
    line_pt_dists = dist_num / dist_denom
    return line_pt_dists


def compute_inliers(raw_pts1, raw_pts2, F_mat):
    raw_pts1_homo = utils.homogenize_points(raw_pts1)
    raw_pts2_homo = utils.homogenize_points(raw_pts2)

    lines_prime = (F_mat @ raw_pts1_homo.T).T  # l_prime = F * x
    img2_dist = _line_point_distances(lines_prime, raw_pts2_homo)

    lines = (F_mat.T @ raw_pts2_homo.T).T  # l = F^T * x_prime
    img1_dist = _line_point_distances(lines, raw_pts1_homo)

    pt_line_distances = (img1_dist + img2_dist) / 2
    inliers_indices = pt_line_distances < ERROR_TOL
    return inliers_indices


def comput_F_8_pt(pts1, pts2):
    # step 1: Normalize data via a similarity transform
    T1, pts1_norm = utils.sim_normalized_points(pts1)
    T2, pts2_norm = utils.sim_normalized_points(pts2)

    # step 2: SVD using constraint matrix from 8 correspondences
    A = np.empty((len(pts1_norm), 9))
    A[:, 0] = pts2_norm[:, 0] * pts1_norm[:, 0]
    A[:, 1] = pts2_norm[:, 0] * pts1_norm[:, 1]
    A[:, 2] = pts2_norm[:, 0] * pts1_norm[:, 2]
    A[:, 3] = pts2_norm[:, 1] * pts1_norm[:, 0]
    A[:, 4] = pts2_norm[:, 1] * pts1_norm[:, 1]
    A[:, 5] = pts2_norm[:, 1] * pts1_norm[:, 2]
    A[:, 6] = pts2_norm[:, 2] * pts1_norm[:, 0]
    A[:, 7] = pts2_norm[:, 2] * pts1_norm[:, 1]
    A[:, 8] = pts2_norm[:, 2] * pts1_norm[:, 2]

    u, s, v_t = np.linalg.svd(A)
    F_mat = v_t[-1].reshape(3, 3)

    # step 3: Project F to rank 2 matrix
    u, d, v_t = np.linalg.svd(F_mat)
    d[-1] = 0
    F_mat_reduced = u @ np.diag(d) @ v_t

    # Step 4: Transform solution to pixel space
    F_mat_denormed = T2.T @ F_mat_reduced @ T1
    F_mat_denormed /= F_mat_denormed[-1, -1]
    return F_mat_denormed


def comput_F_ransac(file_name, roi, num_pts=8):
    # load required data
    img1_path = f"{IN_DIR}/{file_name}/front.jpg"
    img2_path = f"{IN_DIR}/{file_name}/back.jpg"
    raw_corresp_path = f"{IN_DIR}/{file_name}/correspondences.npy"

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise ValueError(f"Invalid path for the input image pair")
    raw_correspondences = utils.read_data(raw_corresp_path)

    # if len(raw_correspondences) >  40000:
    #     nn = np.random.choice(len(raw_correspondences), 40000)
    #     raw_correspondences = raw_correspondences[nn]

    raw_pts1, raw_pts2 = raw_correspondences[:, :2], raw_correspondences[:, 2:]
    raw_pts1 = raw_pts1[:, ::-1]
    raw_pts2 = raw_pts2[:, ::-1]

    # perform RANSAC to find the best F matrix
    max_inliers = -1
    max_inliers_indices = None
    inlier_logger = np.empty(NUM_ITERS)
    for i in tqdm(range(NUM_ITERS)):
        # Step 1: select N random points
        indices = np.random.choice(len(raw_pts1), num_pts)
        pts_subset1 = raw_pts1[indices]
        pts_subset2 = raw_pts2[indices]

        # Step 2: Fit epipolar geometry to the subset
        F_mat = comput_F_8_pt(pts_subset1, pts_subset2)

        # Step 3: Count the number of inliers
        curr_inliers_indices = compute_inliers(raw_pts1, raw_pts2, F_mat)
        if sum(curr_inliers_indices) > max_inliers:
            max_inliers = sum(curr_inliers_indices)
            max_inliers_indices = curr_inliers_indices
        inlier_logger[i] = max_inliers / len(raw_pts1)

    # Step 4: Recompute F matrix based on all inliers
    print(
        f'Total inliers for img "{file_name}" is {round(100 * sum(max_inliers_indices)/ len(raw_pts1))}%'
    )
    pts_inliers1 = raw_pts1[max_inliers_indices]
    pts_inliers2 = raw_pts2[max_inliers_indices]
    F_mat_best = comput_F_8_pt(pts_inliers1, pts_inliers2)

    print(f"\nBest F matrix for {file_name} in {num_pts}-pt algorithm:")
    np.savetxt(sys.stdout, F_mat_best, "%0.8f")
    # print(utils.bmatrix(F_mat_best))

    # plot using the best F matrix
    # """
    indices = np.random.choice(len(raw_pts1), num_pts)
    pts1 = raw_pts1[indices]
    pts2 = raw_pts2[indices]
    # """

    # pts1, pts2 = choose_roi_points(raw_pts1, raw_pts2, roi=roi, num_pts=num_pts)
    lines_prime = (F_mat_best @ utils.homogenize_points(pts1).T).T  # l_prime = F * x

    pts_colors = np.random.choice(256, size=(len(pts1), 3)).astype(int)
    view_1_pts = utils.annotate_img_pts(np.copy(img1), pts1, pts_colors, radius=20)
    view_2_pts = utils.annotate_img_pts(np.copy(img2), pts2, pts_colors, radius=20)
    view_2_lines = utils.annotate_img_lines(view_2_pts, lines_prime, pts_colors)

    out_path = f"{OUT_DIR}/{file_name}.jpg"
    images = [view_1_pts, view_2_lines]
    titles = ["View #1 (chosen pts)", "View #2 (epipolar lines)"]
    utils.save_images(
        images,
        titles,
        save_path=out_path,
        size=(1, 2) if img1.shape[0] > img1.shape[1] else (2, 1),
        fig_w=10,
    )
    utils.save_ransac_plot(
        inlier_logger, file_name, num_pts, save_path=f"{OUT_DIR}/{file_name}_plot.jpg"
    )
    np.save(f"{OUT_DIR}/{file_name}_f_mat.npy", F_mat_best)

    return F_mat_best


def choose_roi_points(raw_pts1, raw_pts2, roi, num_pts=8):
    # roi is [x1, y1, x2, y2]
    # subset = np.array([x for x in img_pts if _of_interest(x, roi)], dtype=img_pts.dtype)

    # subset = np.array([x for x in img_pts if _of_interest(x, roi)]]
    is_valid = np.array([_of_interest(x, roi) for x in raw_pts1])
    subset_indices = np.where(is_valid)[0]

    indices = np.random.choice(len(subset_indices), num_pts)
    pts1 = raw_pts1[indices]
    pts2 = raw_pts2[indices]
    return pts1, pts2


def _of_interest(x, roi):
    x_pt, y_pt = x
    x1, y1, x2, y2 = roi
    if x1 <= x_pt <= x2 and y1 <= y_pt <= y2:
        return True
    return False


if __name__ == "__main__":
    dir_rois = [
        ("messi", [0, 0, 0, 0]),
        ("kobe-dwade", [0, 0, 0, 0]),
        ("gwhunting", [0, 0, 0, 0]),
    ]
    for file_name, roi in dir_rois[:1]:
        print("\n\n", "*" * 10, file_name, "*" * 10)

        F_mat = comput_F_ransac(file_name, roi=None)

    print("\n\nexecution done")

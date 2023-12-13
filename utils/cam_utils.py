import pathlib

import cv2
import matplotlib.pyplot as plt
import numpy as np


def find_best_F_matrix(F_candidates, raw_pts1, raw_pts2):
    indices = np.random.choice(len(raw_pts1), 7)
    extra_pts_in_img1 = homogenize_points(raw_pts1[indices])
    extra_pts_in_img2 = homogenize_points(raw_pts2[indices])

    epipole_errors = []
    for F_cand in F_candidates:
        lines_prime = (F_cand @ extra_pts_in_img1.T).T  # l_prime = F * x

        # find the epipolar points as points of intersections (subset of possible candidates)
        pts_of_intersections = []
        for i in range(len(lines_prime) - 1):
            line_1 = lines_prime[i]
            line_2 = lines_prime[i + 1]
            pt_of_intersection = np.cross(line_1, line_2)
            pts_of_intersections.append(pt_of_intersection)

        pts_of_intersections = np.array(pts_of_intersections)
        raw_epipole_error = np.mean(
            np.std(pts_of_intersections, axis=0)
        )  # mean std deviation across all coeffs
        epipole_errors.append(raw_epipole_error)

    # raw_epipole_error should be zero for the ideal candidate
    best_F_mat_index = np.argmin(epipole_errors)
    return best_F_mat_index


def solve_optimal_lambda_7_pt(F1_mat, F2_mat, T1, T2, raw_pts1, raw_pts2, rtol=1e-8):
    det_fun = lambda l: np.linalg.det(l * F1_mat + (1 - l) * F2_mat)

    a = np.zeros(4)
    # evaluate the determinant at different values to get the coefficients of lambda
    a[3] = det_fun(0)
    a[1] = ((det_fun(1) + det_fun(-1)) / 2) - a[3]
    a[0] = (det_fun(2) - 2 * a[1] + a[3] - 2 * det_fun(1)) / 6
    a[2] = ((det_fun(1) - det_fun(-1)) / 2) - a[0]
    all_roots = np.roots(a)

    F_candidates = []
    for root in all_roots:
        if np.isreal(root) or abs(np.imag(root)) < rtol:  # it's a real root
            lambda_coeff = np.real(root)
            F_mat = lambda_coeff * F1_mat + (1 - lambda_coeff) * F2_mat
            F_mat_denormed = T2.T @ F_mat @ T1  # Transform solution to pixel space
            F_mat_denormed /= F_mat_denormed[-1, -1]
            F_candidates.append(F_mat_denormed)

    if len(F_candidates) == 1:
        # print('Got only one valid fundamental matrix so choosing that')
        best_F_mat_index = 0
    else:
        best_F_mat_index = find_best_F_matrix(F_candidates, raw_pts1, raw_pts2)

    return F_candidates[best_F_mat_index]


def compute_E_from_F(F_mat, K1, K2):
    E_mat = K2.T @ F_mat @ K1
    return E_mat


def annotate_img_pts(img, points, colors, radius=10):
    if isinstance(colors, np.ndarray):
        colors = colors.tolist()

    for pt, color in zip(points, colors):
        cv2.circle(
            img, pt.astype(int), radius=radius, color=color, thickness=cv2.FILLED
        )
    return img


def annotate_img_lines(img, lines_prime, colors, thickness=None):
    img_h, img_w = img.shape[:2]
    if thickness is None:
        thickness = img_w // 100
    if isinstance(colors, np.ndarray):
        colors = colors.tolist()

    for (l_1, l_2, l_3), color in zip(lines_prime, colors):
        pt1 = (0, int(-l_3 / l_2))
        pt2 = (img_w, int(-(l_3 + l_1 * img_w) / l_2))
        cv2.line(img, pt1, pt2, color, thickness=thickness)
    return img


def save_ransac_plot(inlier_logger, img_name, num_pts, save_path):
    plt.plot(inlier_logger * 100, "-", color="g")
    plt.title(f"{num_pts}-pt algorithm on {img_name}", fontweight="bold")
    plt.xlabel("Iterations")
    plt.ylabel("Max Inlier ratio")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def save_images(images, titles, save_path, size=(2, 2), fig_w=10, fig_h=10):
    _, axes = plt.subplots(*size, figsize=(fig_w, fig_h), constrained_layout=True)

    r, c = size
    for i in range(r):
        for j in range(c):
            if r == 1 or c == 1:
                ax = axes[i + j]
            else:
                ax = axes[i, j]
            ax.set_title(titles[i * c + j], fontweight="bold")
            ax.imshow(cv2.cvtColor(images[i * c + j], cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])

    plt.savefig(save_path)
    plt.close()


def read_data(path):
    file_extension = pathlib.Path(path).suffix

    if file_extension == ".txt":
        data = np.loadtxt(path)
    elif file_extension == ".npy" or file_extension == ".npz":
        data = np.load(path)
    else:
        raise ValueError(f"unsupported file type for{path}")
    return data


def homogenize_points(pts):
    if pts.shape[1] == 2:
        return np.column_stack([pts, np.ones(len(pts))])  # [n x 2] -> [n x 3]
    elif pts.shape[1] == 3:
        return pts
    else:
        raise ValueError(
            f"Invalid input points shape while homogenizing: array of shape {pts.shape}"
        )


def sim_normalized_points(pts):
    pts_homo = homogenize_points(pts)

    # from lecture 11 slide 20
    # https://www.dropbox.com/scl/fi/6ssokgdylwtjs11qy5ajy/L11_Two_view_Calibration.pdf?rlkey=qqug6w22lpx1gig6mp7kptwrw&dl=0
    x_centroid, y_centroid = np.mean(pts, axis=0)
    d_avg = np.mean(np.linalg.norm(pts - np.array([x_centroid, y_centroid]), axis=1))
    s = np.sqrt(2) / d_avg

    T_mat = np.array([[s, 0, -s * x_centroid], [0, s, -s * y_centroid], [0, 0, 1]])

    normalized_pts = (T_mat @ pts_homo.T).T
    return T_mat, normalized_pts


def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError("bmatrix can at most display two dimensions")
    lines = str(a).replace("[", "").replace("]", "").splitlines()
    rv = [r"\begin{bmatrix}"]
    rv += ["  " + " & ".join(l.split()) + r"\\" for l in lines]
    rv += [r"\end{bmatrix}"]
    return "\n".join(rv)

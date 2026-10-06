"""Faster RANSAC for the fundamental matrix, built on fun_calc/fun_compute's own functions.

Same algorithm as fun_compute.comput_F_ransac (random 8-point fits, inliers by average
point-to-epipolar-line distance, final 8-point refit on all inliers), with two speedups:
  * each hypothesis is scored on a fixed random subset of the correspondences, not all of them;
  * fewer iterations (the number needed depends on the inlier ratio, not on how many points exist).
The final refit uses every inlier among ALL correspondences. Also runs OpenCV's MAGSAC as a check.

Usage: python run_ransac.py <scene> [error_tol_px=7] [iters=5000] [subset=5000]
Reads data/pose/<scene>/correspondences.npy (rows: row1, col1, row2, col2).
Writes data/pose/out/<scene>_f_mat.npy, <scene>_f_mat_magsac.npy and <scene>_inliers.npy.
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "fun_calc")
import fun_compute  # noqa: E402

scene = sys.argv[1]
tol = float(sys.argv[2]) if len(sys.argv) > 2 else 7.0
iters = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
subset = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
fun_compute.ERROR_TOL = tol
rng = np.random.default_rng(0)
np.random.seed(0)

corr = np.load(f"data/pose/{scene}/correspondences.npy")
pts1 = corr[:, :2][:, ::-1].astype(float)  # (x, y), as in fun_compute
pts2 = corr[:, 2:][:, ::-1].astype(float)
sub = rng.choice(len(pts1), min(subset, len(pts1)), replace=False)
s1, s2 = pts1[sub], pts2[sub]

t0 = time.time()
best, best_F = -1, None
for _ in range(iters):
    idx = np.random.choice(len(s1), 8)
    F = fun_compute.comput_F_8_pt(s1[idx], s2[idx])
    n = fun_compute.compute_inliers(s1, s2, F).sum()
    if n > best:
        best, best_F = n, F
inl = fun_compute.compute_inliers(pts1, pts2, best_F)  # score the winner on all points
F_ours = fun_compute.comput_F_8_pt(pts1[inl], pts2[inl])
inl_ours = fun_compute.compute_inliers(pts1, pts2, F_ours)
t_ours = time.time() - t0

t0 = time.time()
F_cv, mask = cv2.findFundamentalMat(pts1, pts2, cv2.USAC_MAGSAC, tol, 0.999, 10000)
t_cv = time.time() - t0
F_cv = F_cv / F_cv[2, 2]
inl_cv = fun_compute.compute_inliers(pts1, pts2, F_cv)


def mean_err(F, m):
    h1, h2 = np.c_[pts1[m], np.ones(m.sum())], np.c_[pts2[m], np.ones(m.sum())]
    l2, l1 = (F @ h1.T).T, (F.T @ h2.T).T
    d = (np.abs((l2 * h2).sum(1)) / np.linalg.norm(l2[:, :2], axis=1)
         + np.abs((l1 * h1).sum(1)) / np.linalg.norm(l1[:, :2], axis=1)) / 2
    return d.mean()


print(f"{len(pts1)} correspondences, tolerance {tol}px")
print(f"ours   ({iters} iters, {len(sub)}-pt scoring): {inl_ours.mean()*100:5.1f}% inliers, "
      f"mean epipolar error {mean_err(F_ours, inl_ours):.2f}px, {t_ours:.1f}s")
print(f"MAGSAC (OpenCV):                         {inl_cv.mean()*100:5.1f}% inliers, "
      f"mean epipolar error {mean_err(F_cv, inl_cv):.2f}px, {t_cv:.1f}s")
print(f"inlier sets agree on {(inl_ours == inl_cv).mean()*100:.1f}% of points")
np.save(f"data/pose/out/{scene}_f_mat.npy", F_ours)
np.save(f"data/pose/out/{scene}_f_mat_magsac.npy", F_cv)
np.save(f"data/pose/out/{scene}_inliers.npy", inl_ours)

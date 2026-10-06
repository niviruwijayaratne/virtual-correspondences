"""Compare independent estimates of the camera-2 pose relative to camera 1.

  A. Fundamental matrix from run_ransac.py -> E = K2^T F K1 -> recoverPose   (if run_ransac.py was run)
  B. PnP: DensePose maps each image-2 pixel to a point on image 1's PARE mesh (placed in camera 1's
     frame), giving 2D(image 2) <-> 3D(camera 1) matches; solvePnPRansac gives camera 2's pose.
  C. Body orientation only: rotation between the two PARE meshes (each in its own camera frame).
  D. Essential matrix estimated directly (calibrated 5-point RANSAC) -> recoverPose.
Each pose is also scored by the share of virtual correspondences within 7px of its epipolar lines.

Usage: python diag_pose.py [focal_px | f1,f2] [scene folder1 folder2]   (defaults: 1600, apartment pair)
"""
import os
import sys

import cv2
import numpy as np
import torch
from mmhuman3d.utils.demo_utils import convert_crop_cam_to_orig_img

import utils.utils as utils

sys.path.insert(0, "fun_calc")
import fun_compute  # noqa: E402

focals = [float(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1600.0]
f1, f2 = (focals * 2)[:2]
SCENE, FOLDER1, FOLDER2 = sys.argv[2:5] if len(sys.argv) > 4 else ("apartment", "single_view2", "single_view1")
H1, W1 = cv2.imread(f"data/test/{FOLDER1}/image.jpg").shape[:2]
H2, W2 = cv2.imread(f"data/test/{FOLDER2}/image.jpg").shape[:2]
K1 = np.array([[f1, 0, W1 / 2], [0, f1, H1 / 2], [0, 0, 1]])
K2 = np.array([[f2, 0, W2 / 2], [0, f2, H2 / 2], [0, 0, 1]])
fun_compute.ERROR_TOL = 7.0


def mesh_in_camera(folder, f, w, h):
    res = np.load(f"data/outputs/{folder}/inference_result.npz", allow_pickle=True)
    v, cam, bbox = np.asarray(res["verts"])[0], np.asarray(res["pred_cams"])[0], np.asarray(res["bboxes_xyxy"])[0][:4]
    sx, sy, tx, ty = convert_crop_cam_to_orig_img(cam[None], bbox[None], w, h, 1.0, 1.25, "xyxy")[0]
    return v + np.array([tx, ty, 2 * f / (w * sx)])


def angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


m1, m2 = mesh_in_camera(FOLDER1, f1, W1, H1), mesh_in_camera(FOLDER2, f2, W2, H2)
corr = np.load(f"data/pose/{SCENE}/correspondences.npy")
p1, p2 = corr[:, :2][:, ::-1].astype(float), corr[:, 2:][:, ::-1].astype(float)
n1 = cv2.undistortPoints(p1[:, None], K1, None)[:, 0]
n2 = cv2.undistortPoints(p2[:, None], K2, None)[:, 0]


def scaled(R, t):
    t = t.ravel()
    return R, t * float(t @ (m2.mean(0) - R @ m1.mean(0)))


def report(name, R, t):
    """R, t map camera-1 coordinates to camera 2: X2 = R X1 + t."""
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Fp = np.linalg.inv(K2).T @ tx @ R @ np.linalg.inv(K1)
    share = fun_compute.compute_inliers(p1, p2, Fp / Fp[2, 2]).mean() * 100
    look2 = R.T @ np.array([0, 0, 1.0])
    yaw = np.degrees(np.arctan2(look2[0], look2[2]))
    print(f"{name:32s} rotation {angle(R):6.1f}°  cam2 yaw {yaw:7.1f}°  cam2 centre {np.round(-R.T @ t, 2)}  "
          f"explains {share:4.1f}%")


poses = {}
if os.path.exists(f"data/pose/out/{SCENE}_f_mat.npy"):
    F = np.load(f"data/pose/out/{SCENE}_f_mat.npy")
    inl = np.load(f"data/pose/out/{SCENE}_inliers.npy")
    _, Ra, ta, _ = cv2.recoverPose(K2.T @ F @ K1, n1[inl], n2[inl], np.eye(3))
    poses["A. fundamental matrix"] = scaled(Ra, ta)

# B. PnP: image-2 pixels <-> points on mesh 1
dp = torch.load(open(f"data/outputs/{FOLDER2}/densepose.pt", "rb"))[0]
box, d = dp["pred_boxes_XYXY"][0].numpy(), dp["pred_densepose"][0]
lab, uv = d.labels.cpu().numpy(), d.uv.cpu().numpy()
ys, xs = np.nonzero(lab)
keep = np.random.default_rng(0).choice(len(ys), min(6000, len(ys)), replace=False)
sub = np.zeros_like(lab)
sub[ys[keep], xs[keep]] = lab[ys[keep], xs[keep]]
xyz, pix = utils.iuv_to_xyz(np.dstack([sub, uv[0], uv[1]]).astype(np.float32), smpl_vertices=m1)
img2_xy = np.c_[pix[:, 1] + box[0], pix[:, 0] + box[1]].astype(np.float64)
ok, rvec, tvec, inl_pnp = cv2.solvePnPRansac(xyz.astype(np.float64), img2_xy, K2, None, reprojectionError=12.0,
                                              iterationsCount=5000, flags=cv2.SOLVEPNP_EPNP)
poses[f"B. PnP ({len(inl_pnp)}/{len(xyz)} inl.)"] = (cv2.Rodrigues(rvec)[0], tvec.ravel())

# C. Body orientations only: m2 ≈ R m1 + t
A0, B0 = m1 - m1.mean(0), m2 - m2.mean(0)
U, _, Vt = np.linalg.svd(A0.T @ B0)
Rc = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
poses["C. PARE body orientations"] = (Rc, m2.mean(0) - Rc @ m1.mean(0))

# D. Essential matrix directly
Ed, emask = cv2.findEssentialMat(n1, n2, np.eye(3), cv2.RANSAC, 0.999, 7.0 / np.mean([f1, f2]), 20000)
_, Rd, td, _ = cv2.recoverPose(Ed, n1[emask.ravel() > 0], n2[emask.ravel() > 0], np.eye(3))
poses["D. essential matrix (5-point)"] = scaled(Rd, td)

for name, (R, t) in poses.items():
    report(name, R, t)

# E. Same correspondences, but image-1 points from a PINHOLE projection of mesh 1 (camera K1, mesh placed at
#    its perspective depth) instead of the pipeline's weak-perspective projection. Uses the PnP sample above:
#    image-2 pixels img2_xy <-> mesh-1 points xyz (camera 1 frame).
q1 = (K1 @ xyz.T).T
q1 = q1[:, :2] / q1[:, 2:3]
m1n = cv2.undistortPoints(q1[:, None], K1, None)[:, 0]
m2n = cv2.undistortPoints(img2_xy[:, None], K2, None)[:, 0]
Ee, eem = cv2.findEssentialMat(m1n, m2n, np.eye(3), cv2.RANSAC, 0.999, 7.0 / np.mean([f1, f2]), 20000)
_, Re, te, _ = cv2.recoverPose(Ee, m1n[eem.ravel() > 0], m2n[eem.ravel() > 0], np.eye(3))
Re, te = scaled(Re, te)
look2 = Re.T @ np.array([0, 0, 1.0])
print(f"{'E. essential, pinhole image-1 pts':32s} rotation {angle(Re):6.1f}°  cam2 yaw {np.degrees(np.arctan2(look2[0], look2[2])):7.1f}°  "
      f"cam2 centre {np.round(-Re.T @ te, 2)}  inliers {(eem > 0).mean()*100:4.1f}% of {len(xyz)}")

# Planarity check: does one homography explain the correspondences about as well as the essential matrix?
Hm, hmask = cv2.findHomography(p1, p2, cv2.RANSAC, 7.0, maxIters=5000)
print(f"PLANARITY homography inliers {(hmask > 0).mean()*100:4.1f}% vs essential matrix inliers {(emask > 0).mean()*100:4.1f}%")

"""Virtual correspondences with a pinhole projection into image 1.

main.py projects the mesh points into image 1 with PARE's weak-perspective camera. Here the PARE mesh is
placed at its perspective depth in camera 1's frame (as in estimate_pose.py) and projected with the
pinhole intrinsics K1 instead. Every DensePose pixel of the person in image 2 is mapped to a point on the
mesh (same IUV -> XYZ step as main.py), giving image-1 pixel <-> image-2 pixel pairs.

Usage: python make_pinhole_correspondences.py <scene> <image1_folder> <image2_folder> <focal_px | f1,f2> [body=camerahmr]
Writes data/pose/<scene>/correspondences_pinhole.npy with rows (row1, col1, row2, col2), like main.py.
"""
import os
import sys

import cv2
import numpy as np
import torch
from mmhuman3d.utils.demo_utils import convert_crop_cam_to_orig_img

import utils.utils as utils

scene, folder1, folder2, focal_arg = sys.argv[1:5]
body = next((a.split("=", 1)[1] for a in sys.argv[5:] if a.startswith("body=")), "pare")
f1 = ([float(v) for v in focal_arg.split(",")] * 2)[0]
H1, W1 = cv2.imread(f"data/test/{folder1}/image.jpg").shape[:2]
K1 = np.array([[f1, 0, W1 / 2], [0, f1, H1 / 2], [0, 0, 1]])

res = np.load(f"data/outputs/{folder1}/inference_result.npz", allow_pickle=True)
v, cam, bbox = np.asarray(res["verts"])[0], np.asarray(res["pred_cams"])[0], np.asarray(res["bboxes_xyxy"])[0][:4]
sx, sy, tx, ty = convert_crop_cam_to_orig_img(cam[None], bbox[None], W1, H1, 1.0, 1.25, "xyxy")[0]
mesh1 = v + np.array([tx, ty, 2 * f1 / (W1 * sx)])
if body == "camerahmr":  # CameraHMR's body is already in camera 1's frame, at the depth implied by f1
    mesh1 = np.load(f"data/outputs_camhmr/{folder1}/body.npz")["verts_cam"].astype(float)

dp = torch.load(open(f"data/outputs/{folder2}/densepose.pt", "rb"))[0]
box, d = dp["pred_boxes_XYXY"][0].numpy(), dp["pred_densepose"][0]
lab, uv = d.labels.cpu().numpy(), d.uv.cpu().numpy()
xyz, pix = utils.iuv_to_xyz(np.dstack([lab, uv[0], uv[1]]).astype(np.float32), smpl_vertices=mesh1)
img2_rc = np.c_[pix[:, 0] + box[1], pix[:, 1] + box[0]]

q = (K1 @ xyz.T).T
img1_rc = np.c_[q[:, 1] / q[:, 2], q[:, 0] / q[:, 2]]
inside = (img1_rc[:, 0] >= 0) & (img1_rc[:, 0] < H1) & (img1_rc[:, 1] >= 0) & (img1_rc[:, 1] < W1)
corr = np.round(np.c_[img1_rc, img2_rc][inside]).astype(int)

os.makedirs(f"data/pose/{scene}", exist_ok=True)
name = "correspondences_camhmr.npy" if body == "camerahmr" else "correspondences_pinhole.npy"
np.save(f"data/pose/{scene}/{name}", corr)
print(scene, name, corr.shape)

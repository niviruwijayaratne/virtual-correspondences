"""Relative camera pose from the virtual correspondences, a Figure-8-style mesh
overlay, and a plot of both camera frustums.

Usage: python estimate_pose.py <scene> <image1_folder> <image2_folder> [focal_px | f1,f2] [method=essential|ours|magsac]

  focal_px: one focal length for both cameras, or "f1,f2" when the photos come from different cameras.

  data/pose/out/<scene>_f_mat.npy, <scene>_inliers.npy  from run_ransac.py
  data/pose/<scene>/correspondences.npy                  rows: row1, col1, row2, col2
  data/outputs/<folder>/inference_result.npz             PARE output for each image
  data/test/<folder>/image.jpg                           the input images

Steps:
  1. Essential matrix with an assumed pinhole K (focal length focal_px, principal point at the
     centre; the images carry no camera metadata). Default "essential": estimated directly with
     calibrated 5-point RANSAC (cv2.findEssentialMat, 7px). "ours"/"magsac": E = K^T F K from
     run_ransac.py's fundamental matrix. An unconstrained F can fit the correspondences while not
     being consistent with the assumed K, and decomposing it then gives a wrong pose (on the
     apartment pair it gave 174 deg vs ~145 deg from PnP; see diag_pose.py), so "essential" is the
     default. cv2.recoverPose gives R, t with X2 = R X1 + t, |t| = 1.
  2. Each PARE mesh is placed in its own camera frame: the weak-perspective crop camera is
     converted to the full image (mmhuman3d's convert_crop_cam_to_orig_img) and to a perspective
     depth tz = 2 f / (W * sx). This puts the person at a metric distance from each camera.
  3. The unknown translation scale s is set so the image-2 mesh, moved into camera 1's frame,
     lands on the image-1 mesh (centroid match). Camera 2 sits at C2 = -s R^T t.
  4. Check: the rotation that best aligns the two meshes (Procrustes) is an independent estimate
     of the relative rotation; the angle between it and R^T is reported.
Outputs (data/pose/out/): <scene>_pose.json, <scene>_overlay.png, <scene>_frustums.png
"""
import json
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mmhuman3d.utils.demo_utils import convert_crop_cam_to_orig_img  # noqa: E402

scene, folder1, folder2 = sys.argv[1:4]
img1 = cv2.imread(f"data/test/{folder1}/image.jpg")
img2 = cv2.imread(f"data/test/{folder2}/image.jpg")
H, W = img1.shape[:2]
H2, W2 = img2.shape[:2]
focals = [float(v) for v in sys.argv[4].split(",")] if len(sys.argv) > 4 else [float(W)]
f1, f2 = (focals * 2)[:2]
which = sys.argv[5] if len(sys.argv) > 5 else "essential"
K1 = np.array([[f1, 0, W / 2], [0, f1, H / 2], [0, 0, 1]])
K2 = np.array([[f2, 0, W2 / 2], [0, f2, H2 / 2], [0, 0, 1]])
K = K1  # camera 1, used for projecting into image 1

corr = np.load(f"data/pose/{scene}/correspondences.npy")
pts1 = corr[:, :2][:, ::-1].astype(np.float64)  # (x, y)
pts2 = corr[:, 2:][:, ::-1].astype(np.float64)
# Normalized image coordinates, so the two cameras can have different intrinsics
n1 = cv2.undistortPoints(pts1[:, None], K1, None)[:, 0]
n2 = cv2.undistortPoints(pts2[:, None], K2, None)[:, 0]
if which == "essential":
    E, emask = cv2.findEssentialMat(n1, n2, np.eye(3), cv2.RANSAC, 0.999, 7.0 / np.mean([f1, f2]), 20000)
    inl = emask.ravel() > 0
else:
    F = np.load(f"data/pose/out/{scene}_f_mat{'_magsac' if which == 'magsac' else ''}.npy")
    E = K2.T @ F @ K1
    inl = np.load(f"data/pose/out/{scene}_inliers.npy")
if which == "magsac":  # use MAGSAC's own inliers (same tolerance as run_ransac.py's default)
    sys.path.insert(0, "fun_calc")
    import fun_compute  # noqa: E402
    fun_compute.ERROR_TOL = 7.0
    inl = fun_compute.compute_inliers(pts1, pts2, F)


def mesh_in_camera(folder, h, w, f):
    """PARE vertices of person 0 in the camera frame (metres; x right, y down, z forward)."""
    res = np.load(f"data/outputs/{folder}/inference_result.npz", allow_pickle=True)
    verts = np.asarray(res["verts"])[0]
    cam = np.asarray(res["pred_cams"])[0]
    bbox = np.asarray(res["bboxes_xyxy"])[0][:4]
    sx, sy, tx, ty = convert_crop_cam_to_orig_img(cam[None], bbox[None], w, h, 1.0, 1.25, "xyxy")[0]
    tz = 2 * f / (w * sx)  # f is this camera's focal length
    return verts + np.array([tx, ty, tz])


def rot_angle(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


# 1. Pose from the essential matrix
_, R, t, pose_mask = cv2.recoverPose(E, n1[inl], n2[inl], np.eye(3))
t = t.ravel()

# 2-3. Meshes in their camera frames, translation scale from the body
m1 = mesh_in_camera(folder1, *img1.shape[:2], f1)
m2 = mesh_in_camera(folder2, *img2.shape[:2], f2)
c1, c2 = m1.mean(0), m2.mean(0)
s = float(t @ (c2 - R @ c1))  # least-squares scale so R^T (c2 - s t) = c1
m2_in_1 = (R.T @ (m2 - s * t).T).T
C2 = -s * (R.T @ t)  # camera 2 centre in camera 1's frame

# 4. Independent check: rotation aligning the two meshes (vertex i is the same body point)
A, B = m2 - c2, m1 - c1
U, _, Vt = np.linalg.svd(A.T @ B)
D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
R_mesh = Vt.T @ D @ U.T  # m1 ≈ R_mesh m2
angle_vs_mesh = rot_angle(R.T, R_mesh)
centroid_gap = float(np.linalg.norm(m2_in_1.mean(0) - c1))
vertex_gap = float(np.linalg.norm(m2_in_1 - m1, axis=1).mean())

out = {
    "focal_px": [f1, f2], "method": which, "inliers": int(inl.sum()), "of": int(len(inl)),
    "recoverPose_inliers": int((pose_mask > 0).sum()),
    "R": R.round(4).tolist(), "t_unit": t.round(4).tolist(), "scale_m": round(s, 3),
    "camera2_center_in_camera1_m": C2.round(3).tolist(),
    "rotation_between_cameras_deg": round(rot_angle(np.eye(3), R), 1),
    "baseline_m": round(float(np.linalg.norm(C2)), 2),
    "distance_cam1_to_person_m": round(float(np.linalg.norm(c1)), 2),
    "distance_cam2_to_person_m": round(float(np.linalg.norm(c2)), 2),
    "check_rotation_vs_mesh_alignment_deg": round(angle_vs_mesh, 1),
    "check_mean_vertex_gap_after_transfer_m": round(vertex_gap, 3),
}
json.dump(out, open(f"data/pose/out/{scene}_pose.json", "w"), indent=1)
for k, v in out.items():
    if k not in ("R", "t_unit"):
        print(f"{k:42s} {v}")

# Overlay (as in the report's Figure 8): image-2 mesh moved into camera 1 with (R, t, s), projected
uv = (K @ m2_in_1.T).T
uv = uv[:, :2] / uv[:, 2:3]
uv1 = (K @ m1.T).T
uv1 = uv1[:, :2] / uv1[:, 2:3]
over = img1.copy()
blue = np.zeros_like(img1)
for x, y in uv1.astype(int):
    cv2.circle(blue, (x, y), 4, (255, 120, 40), -1)
over = cv2.addWeighted(over, 1.0, blue, 0.55, 0)
for x, y in uv.astype(int):
    cv2.circle(over, (x, y), 2, (40, 40, 230), -1)
cv2.imwrite(f"data/pose/out/{scene}_overlay.png", over)

# Frustum plot: camera 1 frame, shown with y up (y flipped) for readability
CAM_COLORS = {f"Camera 1 ({folder1})": "#2a78d6", f"Camera 2 ({folder2})": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e7e6e2"


def frustum(Rc, C, depth, Kc, w, h):
    """Corners of a frustum for a camera with rotation Rc (world->cam), centre C, intrinsics Kc."""
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    rays = (np.linalg.inv(Kc) @ np.c_[corners, np.ones(4)].T).T * depth
    pts = (Rc.T @ rays.T).T + C
    return C, pts


up = np.array([1, -1, 1])  # flip y so "up" is up in the plot
cams = [(np.eye(3), np.zeros(3), K1, W, H), (R, C2, K2, W2, H2)]
depth = 0.35 * max(np.linalg.norm(C2), 1.0)
body = m1[::7] * up

fig = plt.figure(figsize=(12, 5.4), dpi=150)
fig.patch.set_facecolor("#fcfcfb")
ax3 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2)
for ax in (ax3, ax2):
    ax.set_facecolor("#fcfcfb")

ax3.scatter(body[:, 0], body[:, 2], body[:, 1], s=0.6, c="#9a9893", depthshade=False, label="Body (PARE, image 1)")
ax2.scatter(body[:, 0], body[:, 2], s=0.6, c="#9a9893")
for (Rc, C, Kc, wc, hc), (label, col) in zip(cams, CAM_COLORS.items()):
    apex, cs = frustum(Rc, C, depth, Kc, wc, hc)
    apex, cs = apex * up, cs * up
    for k in range(4):
        ax3.plot(*zip(*[(apex[0], apex[2], apex[1]), (cs[k, 0], cs[k, 2], cs[k, 1])]), color=col, lw=1.6)
        a, b = cs[k], cs[(k + 1) % 4]
        ax3.plot([a[0], b[0]], [a[2], b[2]], [a[1], b[1]], color=col, lw=1.6)
    ax3.scatter([apex[0]], [apex[2]], [apex[1]], color=col, s=30, label=label)
    # top-down: apex and the two outer edges of the frustum footprint
    ax2.plot([apex[0], cs[0, 0]], [apex[2], cs[0, 2]], color=col, lw=1.6)
    ax2.plot([apex[0], cs[1, 0]], [apex[2], cs[1, 2]], color=col, lw=1.6)
    ax2.plot([cs[0, 0], cs[1, 0]], [cs[0, 2], cs[1, 2]], color=col, lw=1.6)
    ax2.scatter([apex[0]], [apex[2]], color=col, s=40, zorder=3)
    ax2.annotate(label, (apex[0], apex[2]), textcoords="offset points", xytext=(8, -12),
                 fontsize=9, color=INK)

allpts = np.vstack([body, np.array([[0, 0, 0], C2 * up])])
mid = allpts.mean(0)
span = (allpts.max(0) - allpts.min(0)).max() / 2 + depth
ax3.set_xlim(mid[0] - span, mid[0] + span)
ax3.set_ylim(mid[2] - span, mid[2] + span)
ax3.set_zlim(mid[1] - span, mid[1] + span)
ax3.set_box_aspect((1, 1, 1))
ax3.view_init(elev=22, azim=-60)
ax3.set_xlabel("x (m)", color=MUTED, fontsize=8)
ax3.set_ylabel("z, depth from camera 1 (m)", color=MUTED, fontsize=8)
ax3.set_zlabel("height (m)", color=MUTED, fontsize=8)
ax3.tick_params(colors=MUTED, labelsize=7)
for a in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
    a.pane.set_facecolor("#fcfcfb")
    a.pane.set_edgecolor(GRID)
    a._axinfo["grid"]["color"] = GRID
ax3.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=INK)
ax3.set_title("3D view", color=INK, fontsize=10, loc="left")

ax2.set_aspect("equal")
ax2.set_xlim(mid[0] - span, mid[0] + span)
ax2.set_ylim(mid[2] - span, mid[2] + span)
ax2.set_xlabel("x (m)", color=MUTED, fontsize=8)
ax2.set_ylabel("z, depth from camera 1 (m)", color=MUTED, fontsize=8)
ax2.tick_params(colors=MUTED, labelsize=7)
ax2.grid(color=GRID, lw=0.6)
for sp in ax2.spines.values():
    sp.set_color(GRID)
ax2.set_title("Top-down view", color=INK, fontsize=10, loc="left")
fig.suptitle(f"Recovered cameras: {out['rotation_between_cameras_deg']}° apart, "
             f"{out['baseline_m']} m baseline (focal lengths {f1:.0f}px, {f2:.0f}px)",
             color=INK, fontsize=11, x=0.02, ha="left")
fig.tight_layout()
fig.savefig(f"data/pose/out/{scene}_frustums.png", facecolor=fig.get_facecolor())
print("wrote", f"data/pose/out/{scene}_overlay.png and {scene}_frustums.png")

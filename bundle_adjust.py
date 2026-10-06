"""Bundle adjustment of the two-camera scene with one shared SMPL body.

Usage: python bundle_adjust.py <scene> <image1_folder> <image2_folder> <focal_px | f1,f2> <out_dir>
                               height=<m> dp1=<DensePose .pt for image 1> [roll0] [pose_weight=<w> | frozen]
                               [up1=x,y,z up2=x,y,z] [upright=<w>] [no_arms]
                               [yaw_starts=<n>] [cam_height=<1|2>:<m>] [body=camerahmr] [corr=<file>] [iters=<n>]

Starts from the pose in render_pose_figure.py (essential matrix from the virtual correspondences, scale from
the PARE meshes, scene scaled so the person has the given height) and jointly refines:
  - camera 2's rotation and translation (camera 1 is fixed at the origin),
  - one shared body: SMPL pose, global orientation and translation in camera 1's frame. The shape (betas)
    stays at PARE's image-1 estimate and the body scale stays fixed, so the person keeps the given height.
Each sampled DensePose pixel names a point on the SMPL surface (a face and barycentric weights); the main
term asks that point to reproject onto its own pixel, in both images, under a robust (Geman-McClure) loss.
A prior keeps the body pose near PARE's image-1 estimate.

Writes <out_dir>/summary.json, result.npz and before/after overlays of the body (red) over each image's
DensePose mask (blue).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

out_dir = sys.argv[5]
dp1_path = next(a.split("=", 1)[1] for a in sys.argv[6:] if a.startswith("dp1="))
_extra_args = list(sys.argv[6:])
# roll0: neither camera is rotated about its optical axis. The true vertical u is estimated (one angle: camera
#        1's tilt up/down; camera 1 has no roll, so u lies in its y-z plane) and camera 2 is parametrised by its
#        heading and tilt relative to u, with zero roll. Heights are then measured along u. pose_weight=<w> sets the body-pose prior weight; pose_weight=frozen keeps PARE's pose.
roll0 = "roll0" in sys.argv[6:]
pose_weight = next((a.split("=", 1)[1] for a in sys.argv[6:] if a.startswith("pose_weight=")), "0.02")
freeze_pose = pose_weight == "frozen"
# up1=x,y,z up2=x,y,z: each camera's measured vertical (vertical_vanishing.py). Camera 2's rotation must map
# camera 1's vertical onto its own, which leaves one free angle (the turn about the vertical); heights use up1.
_ups = {a.split("=", 1)[0]: np.array([float(v) for v in a.split("=", 1)[1].split(",")]) for a in sys.argv[6:]
        if a.startswith(("up1=", "up2="))}
level_mode = len(_ups) == 2
# up1 alone (with roll0): camera 1's vertical is measured; camera 2 has zero roll and a free tilt (use this when
# camera 2's photo has no reliable vertical lines)
up1_only = "up1" in _ups and "up2" not in _ups
# upright=<w>: prior keeping the body's head-to-heel axis along the true vertical (person standing straight)
upright_w = float(next((a.split("=", 1)[1] for a in sys.argv[6:] if a.startswith("upright=")), 0.0))
# no_arms: use only DensePose pixels on the torso, legs, feet and head (parts 1-2, 5-14, 23-24), e.g. when the arms
#          moved between the two photos. DensePose parts 3-4 are the hands and 15-22 the arms.
no_arms = "no_arms" in sys.argv[6:]
# cam_height=<role>:<m>: known height above the ground (the body's lowest point along the vertical) of one camera;
# role "1" is the image-1 (PARE) camera, "2" the other one. Enforced with a stiff quadratic penalty (5 cm scale).
_ch = next((a.split("=", 1)[1] for a in sys.argv[6:] if a.startswith("cam_height=")), None)
cam_height = (int(_ch.split(":")[0]), float(_ch.split(":")[1])) if _ch else None
ARM_PARTS = {3, 4, 15, 16, 17, 18, 19, 20, 21, 22}
# corr=<file in data/pose/<scene>/>: correspondences for the initial essential-matrix pose (default correspondences.npy)
_corr = next((a.split("=", 1)[1] for a in sys.argv[6:] if a.startswith("corr=")), "correspondences.npy")
sys.argv = sys.argv[:5] + ["/tmp/unused", _corr] + [a for a in sys.argv[6:] if a.startswith(("height=", "body="))]
pose_weight = 0.0 if freeze_pose else float(pose_weight)
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_pose_figure.py")).read()
exec(src.split("\ndef draw(theme)")[0])  # K1, K2, R, t, s, m1, m2, img1, img2, folder1, folder2, opts, ...

import smplx  # noqa: E402
import torch  # noqa: E402

import utils.densepose_utils as dp_utils  # noqa: E402

torch.manual_seed(0)
os.makedirs(out_dir, exist_ok=True)
faces = np.load("data/body_models/smpl_faces.npy").astype(int)
DP = dp_utils.DensePoseMethods(Path("./data"))


def densepose_samples(path, n, seed):
    """n person pixels from a DensePose dump: pixel (x, y), SMPL vertex triplet and barycentric weights."""
    d = torch.load(open(path, "rb"))[0]
    box, res = d["pred_boxes_XYXY"][0].numpy(), d["pred_densepose"][0]
    lab, uv = res.labels.cpu().numpy(), res.uv.cpu().numpy()
    ys, xs = np.nonzero(lab)
    if no_arms:
        keep = ~np.isin(lab[ys, xs], list(ARM_PARTS))
        ys, xs = ys[keep], xs[keep]
    pick = np.random.default_rng(seed).choice(len(ys), min(n, len(ys)), replace=False)
    tri, bary, px = [], [], []
    for y, x in zip(ys[pick], xs[pick]):
        fi, b1, b2, b3 = DP.IUV2FBC_fast(lab[y, x], uv[0, y, x], uv[1, y, x])
        tri.append(DP.All_vertices[DP.FacesDensePose[fi]] - 1)
        bary.append([b1, b2, b3])
        px.append([x + box[0], y + box[1]])
    mask = np.zeros(img1.shape[:2] if path == dp1_path else img2.shape[:2], bool)
    hh, ww = min(lab.shape[0], mask.shape[0] - int(box[1])), min(lab.shape[1], mask.shape[1] - int(box[0]))
    mask[int(box[1]):int(box[1]) + hh, int(box[0]):int(box[0]) + ww] = lab[:hh, :ww] > 0
    return np.array(tri, dtype=np.int64), np.array(bary, float), np.array(px, float), mask


N = 3000
tri1, bc1, px1, mask1 = densepose_samples(dp1_path, N, 1)
tri2, bc2, px2, mask2 = densepose_samples(f"data/outputs/{folder2}/densepose.pt", N, 2)
print(f"DensePose samples: image 1 {len(px1)}, image 2 {len(px2)}")

# ---- initial state -------------------------------------------------------------------------------------------
if BODY == "camerahmr":
    _b = np.load(f"data/outputs_camhmr/{folder1}/body.npz")
    sp = {"betas": _b["betas"], "body_pose": _b["body_pose"].reshape(-1), "global_orient": _b["global_orient"]}
    trans1 = _b["transl"].astype(float)
else:
    res1 = np.load(f"data/outputs/{folder1}/inference_result.npz", allow_pickle=True)
    sp = res1["smpl"].item()
    cam, bbox = np.asarray(res1["pred_cams"])[0], np.asarray(res1["bboxes_xyxy"])[0][:4]
    sx_, sy_, tx_, ty_ = convert_crop_cam_to_orig_img(cam[None], bbox[None], W1, H1, 1.0, 1.25, "xyxy")[0]
    trans1 = np.array([tx_, ty_, 2 * f1 / (W1 * sx_)])
smpl = smplx.SMPL(model_path="data/body_models/smpl/SMPL_NEUTRAL.pkl")
betas = torch.tensor(sp["betas"]).float().reshape(1, 10)
v0 = smpl(betas=betas, body_pose=torch.tensor(sp["body_pose"]).float().reshape(1, -1),
          global_orient=torch.tensor(sp["global_orient"]).float().reshape(1, -1)).vertices[0].detach().numpy()
k = float(opts["height"]) / standing_height(folder1) if "height" in opts else 1.0  # same scale as m1
assert np.allclose(k * (v0 + trans1), m1, atol=1e-4), "initial body does not match render_pose_figure.py's mesh"

body_pose = torch.tensor(sp["body_pose"]).float().reshape(1, -1).clone().requires_grad_(True)
global_orient = torch.tensor(sp["global_orient"]).float().reshape(1, -1).clone().requires_grad_(True)
transl = torch.tensor(k * trans1).float().clone().requires_grad_(True)
rvec2 = torch.tensor(cv2.Rodrigues(R)[0].ravel()).float().clone().requires_grad_(True)
t2 = torch.tensor(s * t).float().clone().requires_grad_(True)
pose_init = body_pose.detach().clone()
# camera 2 heading/tilt: R's rows are camera 2's axes in camera 1's frame; camera 1's up is -y
_z = R[2]
yaw2 = torch.tensor(float(np.arctan2(_z[0], _z[2]))).requires_grad_(True)
pitch2 = torch.tensor(float(np.arcsin(-_z[1]))).requires_grad_(True)
cam1_tilt = torch.tensor(0.0).requires_grad_(True)  # camera 1's pitch relative to the true horizontal
use_roll0 = False  # the "before" state is the original (unconstrained) pose
if up1_only:
    _u1 = _ups["up1"] / np.linalg.norm(_ups["up1"])
    u1_t = torch.tensor(_u1).float()
if level_mode:
    _u1, _u2 = (_ups[k] / np.linalg.norm(_ups[k]) for k in ("up1", "up2"))
    _ax = np.cross(_u1, _u2)
    _Q = cv2.Rodrigues((_ax / np.linalg.norm(_ax) * np.arccos(np.clip(_u1 @ _u2, -1, 1))).reshape(3, 1))[0] \
        if np.linalg.norm(_ax) > 1e-9 else np.eye(3)
    Q_t, u1_t = torch.tensor(_Q).float(), torch.tensor(_u1).float()
    # initial turn: the angle about u1 that best matches the essential-matrix rotation
    _cands = np.radians(np.arange(-180, 180, 0.25))
    _err = [np.linalg.norm(R - _Q @ cv2.Rodrigues((_u1 * a).reshape(3, 1))[0]) for a in _cands]
    yaw_v = torch.tensor(float(_cands[int(np.argmin(_err))])).requires_grad_(True)


def vertical():
    """True 'up' in camera 1's frame (camera 1's up is -y when it is level)."""
    if use_roll0 and (level_mode or up1_only):
        return u1_t
    if not use_roll0:
        return torch.tensor([0.0, -1.0, 0.0])
    return torch.stack([torch.zeros(()), -torch.cos(cam1_tilt), torch.sin(cam1_tilt)])

K1t, K2t = torch.tensor(K1).float(), torch.tensor(K2).float()
tri1_t, bc1_t, px1_t = torch.tensor(tri1), torch.tensor(bc1).float(), torch.tensor(px1).float()
tri2_t, bc2_t, px2_t = torch.tensor(tri2), torch.tensor(bc2).float(), torch.tensor(px2).float()


def rodrigues(r):
    th = torch.linalg.norm(r) + 1e-12
    kx = torch.zeros(3, 3)
    a = r / th
    kx = torch.stack([torch.stack([torch.zeros(()), -a[2], a[1]]),
                      torch.stack([a[2], torch.zeros(()), -a[0]]),
                      torch.stack([-a[1], a[0], torch.zeros(())])])
    return torch.eye(3) + torch.sin(th) * kx + (1 - torch.cos(th)) * kx @ kx


def camera2_R():
    if not use_roll0:
        return rodrigues(rvec2)
    if level_mode:
        return Q_t @ rodrigues(u1_t * yaw_v)
    u = vertical()
    right = torch.tensor([1.0, 0.0, 0.0])
    right = right - (right @ u) * u                  # camera 1's x-axis made exactly horizontal
    right = right / torch.linalg.norm(right)
    fwd = torch.linalg.cross(u, right)               # horizontal direction camera 1 faces
    z = torch.cos(pitch2) * torch.sin(yaw2) * right + torch.sin(pitch2) * u + torch.cos(pitch2) * torch.cos(yaw2) * fwd
    x = torch.linalg.cross(z, u)
    x = x / torch.linalg.norm(x)
    return torch.stack([x, torch.linalg.cross(z, x), z])


def body_vertices():
    v = smpl(betas=betas, body_pose=body_pose, global_orient=global_orient).vertices[0]
    return k * v + transl


def project(K, X):
    q = X @ K.T
    return q[:, :2] / q[:, 2:3]


def residuals():
    V = body_vertices()
    X1 = (V[tri1_t] * bc1_t[..., None]).sum(1)
    X2 = (V[tri2_t] * bc2_t[..., None]).sum(1) @ camera2_R().T + t2
    return V, project(K1t, X1) - px1_t, project(K2t, X2) - px2_t


def gm(r, sigma=25.0):  # Geman-McClure on pixel error
    e2 = (r ** 2).sum(1)
    return (e2 / (e2 + sigma ** 2)).mean()


def report(tag):
    with torch.no_grad():
        V, r1, r2 = residuals()
    e1, e2 = r1.norm(dim=1).numpy(), r2.norm(dim=1).numpy()
    return {"tag": tag, "image1_median_px": round(float(np.median(e1)), 1), "image2_median_px": round(float(np.median(e2)), 1),
            "image1_within_20px": round(float((e1 < 20).mean()), 3), "image2_within_20px": round(float((e2 < 20).mean()), 3)}


before = report("before")
print(before)
state0 = {"V": body_vertices().detach().numpy().copy(), "R": camera2_R().detach().numpy().copy(), "t": t2.detach().numpy().copy(),
          "up": vertical().detach().numpy().copy()}
use_roll0 = roll0 or level_mode

# ---- optimise ------------------------------------------------------------------------------------------------
params = [global_orient, transl, t2] + ([yaw_v] if level_mode else ([yaw2, pitch2] if up1_only else [yaw2, pitch2, cam1_tilt])
                                         if roll0 else [rvec2]) + ([] if freeze_pose else [body_pose])
print(f"camera 2 {'turn about the measured vertical only' if level_mode else 'roll fixed at 0' if roll0 else 'rotation free'}; body pose {'frozen' if freeze_pose else f'prior weight {pose_weight}'}")


def total_loss():
    _, r1, r2 = residuals()
    loss = gm(r1) + gm(r2) + pose_weight * ((body_pose - pose_init) ** 2).mean()
    if upright_w:
        Vb = body_vertices()
        axis = Vb[411] - 0.5 * (Vb[3463] + Vb[6863])  # SMPL head-top minus mid-heels
        loss = loss + upright_w * (1 - axis @ vertical() / torch.linalg.norm(axis))
    if cam_height:
        loss = loss + 0.05 * ((camera_height(cam_height[0]) - cam_height[1]) / 0.05) ** 2
    return loss


def camera_height(role):
    """Height of camera `role` (1: image-1 camera at the origin, 2: the other) above the body's lowest point."""
    u = vertical()
    floor = (body_vertices() @ u).min()
    C = torch.zeros(3) if role == 1 else -camera2_R().T @ t2
    return C @ u - floor


ITERS = int(next((a.split("=", 1)[1] for a in _extra_args if a.startswith("iters=")), 1500))


def optimise(iters=None, log=True):
    iters = iters or ITERS
    opt = torch.optim.Adam(params, lr=0.01)
    for it in range(iters):
        opt.zero_grad()
        loss = total_loss()
        loss.backward()
        opt.step()
        if log and (it % 300 == 0 or it == iters - 1):
            print(f"iter {it:4d} loss {loss.item():.4f}")
    with torch.no_grad():
        return float(total_loss())


# yaw_starts=<n> (with measured verticals): try n starting turns about the vertical and keep the best fit. For each
# start, camera 2 is placed so the body's centre sits where PARE puts it in camera 2's frame.
n_starts = int(next((a.split("=", 1)[1] for a in _extra_args if a.startswith("yaw_starts=")), 0))
yaw_param = yaw_v if level_mode else yaw2
if (level_mode or roll0) and n_starts:
    init = [p_.detach().clone() for p_ in params]
    c2_body = torch.tensor(m2.mean(0)).float()
    results = []
    for k_ in range(n_starts):
        for p_, v_ in zip(params, init):
            p_.data.copy_(v_)
        yaw_param.data.fill_(-np.pi + 2 * np.pi * k_ / n_starts)
        with torch.no_grad():
            t2.data.copy_(c2_body - camera2_R() @ body_vertices().mean(0))
        final = optimise(log=False)
        results.append((final, [p_.detach().clone() for p_ in params]))
        print(f"start {np.degrees(-np.pi + 2 * np.pi * k_ / n_starts):7.1f} deg -> loss {final:.4f}, "
              f"turn {np.degrees(float(yaw_param)) % 360:6.1f} deg")
    best = min(results, key=lambda r_: r_[0])
    for p_, v_ in zip(params, best[1]):
        p_.data.copy_(v_)
    print(f"best loss {best[0]:.4f}")
else:
    optimise()
after = report("after")
print(after)
state1 = {"V": body_vertices().detach().numpy().copy(), "R": camera2_R().detach().numpy().copy(), "t": t2.detach().numpy().copy(),
          "up": vertical().detach().numpy().copy()}


# ---- geometry summary ----------------------------------------------------------------------------------------
def geometry(st):
    V, Rb, tb = st["V"], st["R"], st["t"]
    C2b = -Rb.T @ tb
    ang = float(np.degrees(np.arccos(np.clip((np.trace(Rb) - 1) / 2, -1, 1))))
    out = {"rotation_deg": round(ang, 1)}
    # vertical from camera 1 (camera held level) and from the body itself (person standing upright)
    up_cam = st["up"]  # estimated vertical (camera 1's -y when camera 1 is assumed level)
    head, heels = V[411], 0.5 * (V[3463] + V[6863])  # SMPL head-top and heel vertices
    up_body = (head - heels) / np.linalg.norm(head - heels)
    for name, up in (("estimated_vertical", up_cam), ("body_upright", up_body)):
        floor = (V @ up).min()
        centre = V.mean(0)
        flat = lambda p: p - (p @ up) * up  # noqa: E731
        out[name] = {
            "camera1_height_ft": round(float((0 - floor) * FT), 1),
            "camera2_height_ft": round(float((C2b @ up - floor) * FT), 1),
            "person_height_ft": round(float(np.ptp(V @ up) * FT), 2),
            "camera1_to_person_ft": round(float(np.linalg.norm(flat(centre)) * FT), 1),
            "camera2_to_person_ft": round(float(np.linalg.norm(flat(centre - C2b)) * FT), 1),
            "angle_around_person_deg": round(float(np.degrees(np.arccos(np.clip(
                flat(-centre) @ flat(C2b - centre) / np.linalg.norm(flat(-centre)) / np.linalg.norm(flat(C2b - centre)), -1, 1)))), 1),
        }
    out["camera1_tilt_deg"] = round(float(np.degrees(np.arcsin(np.clip(np.array([0, 0, 1.0]) @ up_cam, -1, 1)))), 1)
    out["camera2_roll_deg"] = round(float(np.degrees(np.arcsin(np.clip(Rb[0] @ up_cam, -1, 1)))), 1)
    out["camera2_pitch_deg"] = round(float(np.degrees(np.arcsin(np.clip(Rb[2] @ up_cam, -1, 1)))), 1)
    tilt = np.degrees(np.arccos(np.clip(up_cam @ up_body, -1, 1)))
    out["body_tilt_from_camera1_vertical_deg"] = round(float(tilt), 1)
    return out


from scipy.spatial.transform import Rotation as _Rot  # noqa: E402
_joint_change = [float(np.degrees((_Rot.from_rotvec(a).inv() * _Rot.from_rotvec(b)).magnitude()))
                 for a, b in zip(pose_init.numpy().reshape(23, 3), body_pose.detach().numpy().reshape(23, 3))]
summary = {"before": {**before, **geometry(state0)}, "after": {**after, **geometry(state1)},
           "settings": {"no_arms": no_arms, "cam_height": cam_height, "camera2_roll_fixed_at_0": roll0, "measured_verticals": level_mode, "upright_weight": upright_w, "body_pose": "frozen" if freeze_pose else f"prior weight {pose_weight}"},
           "largest_joint_change_deg": round(max(_joint_change), 1)}
json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=1)
np.savez(f"{out_dir}/result.npz", V_before=state0["V"], R_before=state0["R"], t_before=state0["t"],
         V_after=state1["V"], R_after=state1["R"], t_after=state1["t"], K1=K1, K2=K2, scale_k=k,
         betas=betas.numpy(), body_pose_after=body_pose.detach().numpy().copy(),
         global_orient_after=global_orient.detach().numpy().copy(), transl_after=transl.detach().numpy().copy(),
         up_before=state0["up"], up_after=state1["up"])
print(json.dumps(summary, indent=1))


# ---- overlays ------------------------------------------------------------------------------------------------
def overlay(img, mask, K, X, path):
    h, w = img.shape[:2]
    q = X @ K.T
    uv = q[:, :2] / q[:, 2:3]
    base = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    blue = base.copy()
    blue[mask] = (235, 90, 30)
    out = cv2.addWeighted(base, 0.35, blue, 0.65, 0)
    out[~mask] = base[~mask]
    red = np.zeros((h, w), np.uint8)
    for fc in faces[np.argsort(-X[faces].mean(1)[:, 2])]:
        cv2.fillPoly(red, [np.round(uv[fc]).astype(np.int32)], 255, cv2.LINE_AA)
    a = (red.astype(float) / 255 * 0.85)[..., None]
    out = (out * (1 - a) + np.array([40, 40, 225]) * a).astype(np.uint8)
    cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 88])
    m = red > 0
    return round(float((m & mask).sum() / (m | mask).sum()), 3)


ious = {}
for tag, st in (("before", state0), ("after", state1)):
    ious[f"{tag}_image1_iou"] = overlay(img1, mask1, K1, st["V"], f"{out_dir}/{tag}_image1.jpg")
    ious[f"{tag}_image2_iou"] = overlay(img2, mask2, K2, st["V"] @ st["R"].T + st["t"], f"{out_dir}/{tag}_image2.jpg")
summary["overlap_iou"] = ious
json.dump(summary, open(f"{out_dir}/summary.json", "w"), indent=1)
print("overlap IoU (body vs DensePose mask):", ious)

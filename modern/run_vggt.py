"""Relative camera pose for each image pair with VGGT (feed-forward cameras + point maps), compared with ours.

Usage: python run_vggt.py [inputs_dir] [out_dir]

For each scene in inputs/scenes.json: runs VGGT-1B on (image 1, image 2) in "pad" mode (whole images kept), and
reads camera 2's pose relative to camera 1, VGGT's predicted focal lengths, and camera 1's point map. The scene is
scaled so the person (camera 1's points inside the DensePose mask) is 5'10" tall along camera 1's measured
vertical; camera heights and distances are then measured as in bundle_adjust.py. Writes out/<scene>.json and
out/summary.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

IN = Path(sys.argv[1] if len(sys.argv) > 1 else "inputs")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "out_vggt")
OUT.mkdir(exist_ok=True)
FT, HEIGHT = 3.28084, 1.778
device = "mps" if torch.backends.mps.is_available() else "cpu"

model = VGGT()
from huggingface_hub import hf_hub_download  # noqa: E402
model.load_state_dict(torch.load(hf_hub_download("facebook/VGGT-1B", "model.pt"), map_location="cpu"))
model = model.eval().to(device)


def pad_transform(w, h, target=518):
    """Scale and offsets of VGGT's 'pad' preprocessing: original pixel -> padded 518x518 pixel."""
    if w >= h:
        nw, nh = target, round(h * (target / w) / 14) * 14
    else:
        nh, nw = target, round(w * (target / h) / 14) * 14
    return nw / w, nh / h, (target - nw) // 2, (target - nh) // 2


def angle(Rm):
    return float(np.degrees(np.arccos(np.clip((np.trace(Rm) - 1) / 2, -1, 1))))


summary = []
for sc in json.load(open(IN / "scenes.json")):
    name = sc["scene"]
    paths = [str(IN / f"{name}_1.jpg"), str(IN / f"{name}_2.jpg")]
    images = load_and_preprocess_images(paths, mode="pad").to(device)
    with torch.no_grad():
        pred = model(images[None])
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    extr, intr = extr[0].cpu().numpy().astype(float), intr[0].cpu().numpy().astype(float)
    E1, E2 = extr
    R1, t1, R2, t2 = E1[:, :3], E1[:, 3], E2[:, :3], E2[:, 3]
    Rrel = R2 @ R1.T                    # camera 1 -> camera 2
    trel = t2 - Rrel @ t1
    world_points = pred["world_points"][0, 0].cpu().numpy()   # camera 1's pixels, in the world (= camera 1) frame
    conf = pred["world_points_conf"][0, 0].cpu().numpy()
    # express points in camera 1's frame (VGGT's world is camera 1, but apply E1 in case it is not exactly identity)
    P = world_points.reshape(-1, 3) @ R1.T + t1

    # person points: camera 1's DensePose mask mapped into the padded 518 grid
    w0, h0 = Image.open(paths[0]).size
    sx, sy, ox, oy = pad_transform(w0, h0)
    mask = np.load(IN / f"{name}_mask1.npy")
    ys, xs = np.nonzero(mask)
    u = np.clip(np.round(xs * sx + ox).astype(int), 0, 517)
    v = np.clip(np.round(ys * sy + oy).astype(int), 0, 517)
    sel = np.zeros((518, 518), bool)
    sel[v, u] = True
    sel &= conf > np.percentile(conf[sel], 20)
    person = P.reshape(518, 518, 3)[sel]

    up = np.array(sc["up1"], float)
    up /= np.linalg.norm(up)
    hts = person @ up
    lo, hi = np.percentile(hts, [1, 99])
    k = HEIGHT / (hi - lo)               # metres per VGGT unit
    centre = np.median(person, 0)
    C2 = -Rrel.T @ trel
    flat = lambda p: p - (p @ up) * up   # noqa: E731
    v1, v2 = flat(-centre), flat(C2 - centre)
    around = float(np.degrees(np.arccos(np.clip(v1 @ v2 / np.linalg.norm(v1) / np.linalg.norm(v2), -1, 1))))

    # VGGT's focal lengths, in original-image pixels
    focal = [float(intr[i][0, 0] / sx) for i in range(2)]
    ours = np.load(IN / f"{name}_ours.npz")
    res = {
        "scene": name,
        "rotation_deg": round(angle(Rrel), 1),
        "angle_around_person_deg": round(around, 1),
        "camera1_height_ft": round(float((0 - lo) * k * FT), 1),
        "camera2_height_ft": round(float((C2 @ up - lo) * k * FT), 1),
        "camera1_to_person_ft": round(float(np.linalg.norm(v1) * k * FT), 1),
        "camera2_to_person_ft": round(float(np.linalg.norm(v2) * k * FT), 1),
        "focal_px": [round(f) for f in focal],
        "our_focal_px": [sc["focal1"], sc["focal2"]],
        "rotation_diff_vs_ours_before_deg": round(angle(Rrel @ ours["R_before"].T), 1),
        "rotation_diff_vs_ours_after_deg": round(angle(Rrel @ ours["R_after"].T), 1),
        "person_points": int(sel.sum()),
    }
    json.dump(res, open(OUT / f"{name}.json", "w"), indent=1)
    np.savez(OUT / f"{name}.npz", R=Rrel, t=trel, scale_m_per_unit=k, points=P.reshape(518, 518, 3), conf=conf,
             points2=pred["world_points"][0, 1].cpu().numpy(), conf2=pred["world_points_conf"][0, 1].cpu().numpy(),
             images=images.cpu().numpy())
    summary.append(res)
    print(json.dumps(res))
json.dump(summary, open(OUT / "summary.json", "w"), indent=1)

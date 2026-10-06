"""Run CameraHMR (perspective-camera human mesh recovery) on both photos of each scene.

Usage: python run_camerahmr.py [inputs_dir] [out_dir]

Person boxes come from the DensePose masks in inputs/ (no detector). The focal length passed to CameraHMR is the
one used everywhere else (inputs/scenes.json); CameraHMR's own focal estimate (HumanFoV) is recorded alongside.
Writes out/<scene>_<1|2>.npz with SMPL betas, body_pose (23x3 axis-angle), global_orient (axis-angle), transl
(camera frame, metres) and verts_cam (vertices in the camera frame, metres).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "CameraHMR"))
import os  # noqa: E402

os.chdir(ROOT / "CameraHMR")  # CameraHMR's constants use paths relative to its repo
from core.datasets.dataset import Dataset  # noqa: E402
from core.camerahmr_model import CameraHMR  # noqa: E402
from core.cam_model.fl_net import FLNet  # noqa: E402
from core.constants import CHECKPOINT_PATH, CAM_MODEL_CKPT, SMPL_MODEL_PATH, NUM_BETAS  # noqa: E402


def resize_image(img, target_size):
    """Letterbox to a white target_size square (same as CameraHMR's mesh_estimator.resize_image, which can't be
    imported here because that module also imports detectron2)."""
    height, width = img.shape[:2]
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_width, new_height = target_size, int(target_size / aspect_ratio)
    else:
        new_width, new_height = int(target_size * aspect_ratio), target_size
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    final = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255
    sx, sy = (target_size - new_width) // 2, (target_size - new_height) // 2
    final[sy:sy + new_height, sx:sx + new_width] = resized
    return aspect_ratio, final


import smplx  # noqa: E402
from torchvision.transforms import Normalize  # noqa: E402
from core.constants import IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD  # noqa: E402

IN = (ROOT / (sys.argv[1] if len(sys.argv) > 1 else "inputs")).resolve()
OUT = (ROOT / (sys.argv[2] if len(sys.argv) > 2 else "out_camhmr")).resolve()
OUT.mkdir(exist_ok=True)

model = CameraHMR.load_from_checkpoint(CHECKPOINT_PATH, strict=False, model_type="smpl", map_location="cpu").eval()
fov_model = FLNet()
fov_model.load_state_dict(torch.load(CAM_MODEL_CKPT, map_location="cpu")["state_dict"])
fov_model.eval()
body_model = smplx.SMPLLayer(model_path=SMPL_MODEL_PATH, num_betas=NUM_BETAS)
normalize = Normalize(mean=IMAGE_MEAN, std=IMAGE_STD)


def humanfov_focal(img):
    h, w = img.shape[:2]
    _, small = resize_image(img, IMAGE_SIZE)
    x = normalize(torch.from_numpy(np.transpose(small.astype("float32"), (2, 0, 1)) / 255.0))
    with torch.no_grad():
        fov, _ = fov_model(x[None])
    return float(h / (2 * torch.tan(fov[0, 1] / 2)))


for sc in json.load(open(IN / "scenes.json")):
    for i in (1, 2):
        img = cv2.cvtColor(cv2.imread(str(IN / f"{sc['scene']}_{i}.jpg")), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        f = float(sc[f"focal{i}"])
        cam_int = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float32)
        ys, xs = np.nonzero(np.load(IN / f"{sc['scene']}_mask{i}.npy"))
        box = np.array([[xs.min(), ys.min(), xs.max(), ys.max()]], float)
        centre, scale = (box[:, 2:] + box[:, :2]) / 2, (box[:, 2:] - box[:, :2]) / 200.0
        batch = next(iter(torch.utils.data.DataLoader(Dataset(img, centre, scale, cam_int, False, None), batch_size=1)))
        with torch.no_grad():
            params, pred_cam, _ = model(batch)
            out = body_model(**{k: v.float() for k, v in params.items()})
        # full-image camera translation, as in CameraHMR's mesh_estimator.convert_to_full_img_cam
        s_, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
        bh, bc = batch["box_size"], batch["box_center"]
        transl = torch.stack([tx + 2 * (bc[:, 0] - w / 2) / (s_ * bh), ty + 2 * (bc[:, 1] - h / 2) / (s_ * bh),
                              2 * f / (bh * s_)], -1)[0].numpy()
        verts = out.vertices[0].numpy()
        go = Rotation.from_matrix(params["global_orient"][0, 0].numpy()).as_rotvec()
        bp = Rotation.from_matrix(params["body_pose"][0].numpy().reshape(-1, 3, 3)).as_rotvec()
        np.savez(OUT / f"{sc['scene']}_{i}.npz", betas=params["betas"][0].numpy(), global_orient=go, body_pose=bp,
                 transl=transl, verts_cam=verts + transl, focal=f, focal_humanfov=humanfov_focal(img), box=box[0])
        print(f"{sc['scene']} image {i}: depth {transl[2]:.2f} m, focal used {f:.0f}, HumanFoV estimate "
              f"{humanfov_focal(img):.0f}, body height (rest pose) see betas")

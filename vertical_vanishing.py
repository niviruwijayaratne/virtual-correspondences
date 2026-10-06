"""Estimate each camera's true 'up' direction from vertical lines in the scene (door frames, walls, windows).

Usage: python vertical_vanishing.py <image> <focal_px> [person_mask_densepose.pt] [debug_out.jpg]

Detects line segments, keeps the near-vertical ones away from the person, and fits their common vanishing
point with RANSAC. The vanishing direction d = K^-1 v (sign chosen to point up) gives the camera's roll
(image x-axis tilt, asin(d_x)) and pitch (optical-axis elevation, asin(d_z)) relative to the true vertical.
Prints JSON: up vector in camera coordinates, roll, pitch, number of supporting lines.
"""
import json
import sys

import cv2
import numpy as np

img_path, f = sys.argv[1], float(sys.argv[2])
img = cv2.imread(img_path)
h, w = img.shape[:2]
K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
mask = np.zeros((h, w), bool)
if len(sys.argv) > 3 and sys.argv[3].endswith(".pt"):
    import torch
    d = torch.load(open(sys.argv[3], "rb"))[0]
    box = d["pred_boxes_XYXY"][0].numpy().astype(int)
    mask[max(0, box[1] - 20):box[3] + 20, max(0, box[0] - 20):box[2] + 20] = True

lsd = cv2.createLineSegmentDetector()
segs = lsd.detect(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0].reshape(-1, 4)
keep = []
for x1, y1, x2, y2 in segs:
    L = np.hypot(x2 - x1, y2 - y1)
    ang = np.degrees(np.arctan2(abs(x2 - x1), abs(y2 - y1)))  # 0 = vertical in the image
    mx, my = int((x1 + x2) / 2), int((y1 + y2) / 2)
    if L > 40 and ang < 20 and not mask[min(my, h - 1), min(mx, w - 1)]:
        keep.append([x1, y1, x2, y2])
keep = np.array(keep)
lines = np.cross(np.c_[keep[:, :2], np.ones(len(keep))], np.c_[keep[:, 2:], np.ones(len(keep))])
lines /= np.linalg.norm(lines[:, :2], axis=1, keepdims=True)
lengths = np.hypot(keep[:, 2] - keep[:, 0], keep[:, 3] - keep[:, 1])

# RANSAC over pairs of lines; inlier if the angle between the segment and the direction to the VP is small
rng = np.random.default_rng(0)
best, best_score = None, -1
mids = (keep[:, :2] + keep[:, 2:]) / 2
dirs = (keep[:, 2:] - keep[:, :2]) / lengths[:, None]
def inliers(v):
    to_v = v[:2][None] - mids * v[2] if abs(v[2]) > 1e-12 else np.tile(v[:2], (len(mids), 1))
    to_v = to_v / np.linalg.norm(to_v, axis=1, keepdims=True)
    return np.abs((dirs * to_v).sum(1)) > np.cos(np.radians(1.5))
for _ in range(3000):
    i, j = rng.choice(len(lines), 2, replace=False)
    v = np.cross(lines[i], lines[j])
    if np.linalg.norm(v) < 1e-12:
        continue
    inl = inliers(v)
    score = lengths[inl].sum()
    if score > best_score:
        best, best_score = inl, score
# refine: least squares on inlier lines (weighted by length)
A = lines[best] * lengths[best, None]
v = np.linalg.svd(A)[2][-1]
d = np.linalg.inv(K) @ v
d /= np.linalg.norm(d)
if d[1] > 0:
    d = -d  # point up (image y points down)
out = {"up": d.round(4).tolist(), "roll_deg": round(float(np.degrees(np.arcsin(d[0]))), 2),
       "pitch_deg": round(float(np.degrees(np.arcsin(d[2]))), 2), "lines": int(best.sum()), "candidates": int(len(keep))}
print(json.dumps(out))
if len(sys.argv) > 4:
    dbg = img.copy()
    for (x1, y1, x2, y2), ok in zip(keep, best):
        cv2.line(dbg, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 0) if ok else (0, 0, 255), 3, cv2.LINE_AA)
    cv2.imwrite(sys.argv[4], dbg)

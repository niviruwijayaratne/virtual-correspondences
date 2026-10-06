"""Match two images with SIFT and with SuperGlue, and draw all three match sets (SIFT, SuperGlue, virtual
correspondences) in the same style for the website.

Usage: python run_baselines.py <image1_folder> <image2_folder> <out_dir> [indoor|outdoor] [num_vc_lines] [vc.npy] [swap]

image1 is the PARE image (the one passed as --image1_path to main.py), so the virtual correspondences in
data/outputs/<image1_folder>/correspondences.npy line up with the left image. Writes sift.jpg,
superglue.jpg, vc.jpg and counts.json to <out_dir>, plus pair.jpg and matches.json (the photos without lines, and each
method's lines as coordinates) for the website's interactive match figure.
  - SIFT: OpenCV SIFT, nearest-neighbour matching with Lowe's ratio test (0.75).
  - SuperGlue: SuperPoint + SuperGlue (indoor or outdoor weights to match the scene, match threshold 0.2), images resized to 640 px on
    the long side as in the SuperGlue demo.
"""
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline", "SuperGluePretrainedNetwork"))
from models.matching import Matching  # noqa: E402

folder1, folder2, out_dir = sys.argv[1:4]
weights = sys.argv[4] if len(sys.argv) > 4 else "outdoor"
n_vc = int(sys.argv[5]) if len(sys.argv) > 5 else 150
vc_path = sys.argv[6] if len(sys.argv) > 6 else f"data/outputs/{folder1}/correspondences.npy"
os.makedirs(out_dir, exist_ok=True)
img1 = cv2.imread(f"data/test/{folder1}/image.jpg")
img2 = cv2.imread(f"data/test/{folder2}/image.jpg")
g1, g2 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


def sift_matches():
    sift = cv2.SIFT_create()
    k1, d1 = sift.detectAndCompute(g1, None)
    k2, d2 = sift.detectAndCompute(g2, None)
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    return np.array([[*k1[m.queryIdx].pt, *k2[m.trainIdx].pt] for m in good]).reshape(-1, 4)


def superglue_matches(long_side=640):
    def prep(g):
        s = long_side / max(g.shape)
        r = cv2.resize(g, (round(g.shape[1] * s), round(g.shape[0] * s)), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(r / 255.0).float()[None, None], s

    t1, s1 = prep(g1)
    t2, s2 = prep(g2)
    matching = Matching({"superpoint": {"nms_radius": 4, "keypoint_threshold": 0.005, "max_keypoints": 1024},
                         "superglue": {"weights": weights, "sinkhorn_iterations": 20, "match_threshold": 0.2}}).eval()
    with torch.no_grad():
        pred = matching({"image0": t1, "image1": t2})
    k1, k2 = pred["keypoints0"][0].numpy(), pred["keypoints1"][0].numpy()
    m = pred["matches0"][0].numpy()
    ok = m > -1
    return np.c_[k1[ok] / s1, k2[m[ok]] / s2]


def draw(matches, path, colour=(60, 200, 60)):
    """Side by side at equal height; matches are (x1, y1, x2, y2) in original pixels."""
    h = min(img1.shape[0], img2.shape[0])
    a = cv2.resize(img1, (round(img1.shape[1] * h / img1.shape[0]), h))
    b = cv2.resize(img2, (round(img2.shape[1] * h / img2.shape[0]), h))
    sa, sb = h / img1.shape[0], h / img2.shape[0]
    gap = 12
    canvas = np.full((h, a.shape[1] + gap + b.shape[1], 3), 255, np.uint8)
    canvas[:, :a.shape[1]] = a
    canvas[:, a.shape[1] + gap:] = b
    off = a.shape[1] + gap
    lw = max(2, round(h / 600))
    for x1, y1, x2, y2 in matches:
        p, q = (round(x1 * sa), round(y1 * sa)), (round(x2 * sb) + off, round(y2 * sb))
        cv2.line(canvas, p, q, colour, lw, cv2.LINE_AA)
    for x1, y1, x2, y2 in matches:
        for c in ((round(x1 * sa), round(y1 * sa)), (round(x2 * sb) + off, round(y2 * sb))):
            cv2.circle(canvas, c, lw + 3, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, c, lw + 1, (214, 120, 42), -1, cv2.LINE_AA)
    cv2.imwrite(path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])


sift = sift_matches()
sg = superglue_matches()
vc = np.load(vc_path)  # rows (r1, c1, r2, c2)
vc_xy = vc[:, [1, 0, 3, 2]].astype(float)
vc_draw = vc_xy[np.linspace(0, len(vc_xy) - 1, min(n_vc, len(vc_xy))).astype(int)]

if "swap" in sys.argv[7:]:  # draw image 2 on the left (camera naming where image 2 is camera 1)
    img1, img2 = img2, img1
    sift, sg, vc_draw = (m[:, [2, 3, 0, 1]] for m in (sift, sg, vc_draw))
draw(sift, f"{out_dir}/sift.jpg")
draw(sg, f"{out_dir}/superglue.jpg")
draw(vc_draw, f"{out_dir}/vc.jpg")
counts = {"sift": len(sift), "superglue": len(sg), "vc": len(vc_xy), "vc_drawn": len(vc_draw)}
json.dump(counts, open(f"{out_dir}/counts.json", "w"))


def export_layers():
    """For the website's interactive match figure: pair.jpg (the two photos side by side, no lines, same layout as
    draw()) and matches.json (each method's match lines in pair.jpg pixel coordinates)."""
    h = min(img1.shape[0], img2.shape[0])
    a = cv2.resize(img1, (round(img1.shape[1] * h / img1.shape[0]), h))
    b = cv2.resize(img2, (round(img2.shape[1] * h / img2.shape[0]), h))
    sa, sb, gap = h / img1.shape[0], h / img2.shape[0], 12
    canvas = np.full((h, a.shape[1] + gap + b.shape[1], 3), 255, np.uint8)
    canvas[:, :a.shape[1]] = a
    canvas[:, a.shape[1] + gap:] = b
    off = a.shape[1] + gap
    cv2.imwrite(f"{out_dir}/pair.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    lines = lambda m: [[round(x1 * sa, 1), round(y1 * sa, 1), round(x2 * sb + off, 1), round(y2 * sb, 1)]  # noqa: E731
                       for x1, y1, x2, y2 in m]
    json.dump({"width": int(canvas.shape[1]), "height": int(h), "split": int(a.shape[1]),
               "methods": {"superglue": {"count": len(sg), "lines": lines(sg)},
                           "sift": {"count": len(sift), "lines": lines(sift)},
                           "vc": {"count": len(vc_xy), "lines": lines(vc_draw)}}},
              open(f"{out_dir}/matches.json", "w"), separators=(",", ":"))


export_layers()
print(folder1, folder2, counts)

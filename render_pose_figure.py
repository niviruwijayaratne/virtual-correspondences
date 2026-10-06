"""Render a clean 3D figure of the recovered cameras for the website: both camera frustums with
their photos on the image planes, the PARE body between them, and a faint floor grid.

Usage: python render_pose_figure.py <scene> <image1_folder> <image2_folder> <focal_px | f1,f2> <out_prefix> [corr_file] [image1_densepose.pt] [height=<m>] [ba=<bundle_adjust result.npz>] [swap_labels] [cam1_height=<m>] [body=camerahmr]
Writes <out_prefix>_light.png and <out_prefix>_dark.png. The pose is estimated as in
estimate_pose.py (essential matrix from the virtual correspondences, scale from the PARE meshes).
corr_file defaults to correspondences.npy in data/pose/<scene>/. Given image1_densepose.pt (DensePose run on
image 1), also writes <out_prefix>_projection.jpg: camera 2's mesh projected into image 1 over the person's mask.
"""
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mmhuman3d.utils.demo_utils import convert_crop_cam_to_orig_img  # noqa: E402

scene, folder1, folder2, focal_arg, out_prefix = sys.argv[1:6]
# optional "height=<metres>": rescale the scene so the person (image-1 mesh) has this height
# optional "ba=<result.npz>": use the bundle-adjusted pose and shared body from bundle_adjust.py instead
# optional "cam1_height=<m>": rescale the whole scene so camera 1 (as named on the page) is this high above the
# person's feet; this replaces the person's height as the scale reference
# optional "body=camerahmr": use CameraHMR's perspective-camera bodies (data/outputs_camhmr/<folder>/body.npz)
# instead of PARE's
opts = dict(a.split("=", 1) for a in sys.argv[6:] if a.startswith(("height=", "ba=", "cam1_height=", "body=")))
BODY = opts.get("body", "pare")
# optional "swap_labels": the image-1 (PARE) camera is called "Camera 2" and the other one "Camera 1", with the
# colours swapped to match (the geometry is unchanged)
SWAP = "swap_labels" in sys.argv[6:]
sys.argv = [a for a in sys.argv if not a.startswith(("height=", "ba=", "cam1_height=", "body=")) and a != "swap_labels"]
NAMES = ("Camera 2", "Camera 1") if SWAP else ("Camera 1", "Camera 2")


def cam_colour(i, light):
    """Colour of the camera in role i (0 = image-1/PARE camera): Camera 1 blue, Camera 2 orange."""
    blue, orange = ("#2a78d6" if light else "#3987e5"), ("#eb6834" if light else "#d95926")
    return (orange, blue)[i] if SWAP else (blue, orange)[i]
corr_file = sys.argv[6] if len(sys.argv) > 6 else "correspondences.npy"
img1 = cv2.cvtColor(cv2.imread(f"data/test/{folder1}/image.jpg"), cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread(f"data/test/{folder2}/image.jpg"), cv2.COLOR_BGR2RGB)
(H1, W1), (H2, W2) = img1.shape[:2], img2.shape[:2]
f1, f2 = ([float(v) for v in focal_arg.split(",")] * 2)[:2]
K1 = np.array([[f1, 0, W1 / 2], [0, f1, H1 / 2], [0, 0, 1]])
K2 = np.array([[f2, 0, W2 / 2], [0, f2, H2 / 2], [0, 0, 1]])


def mesh_in_camera(folder, f, w, h):
    if BODY == "camerahmr":  # already in the camera frame, at the depth implied by focal length f
        return np.load(f"data/outputs_camhmr/{folder}/body.npz")["verts_cam"].astype(float)
    res = np.load(f"data/outputs/{folder}/inference_result.npz", allow_pickle=True)
    v, cam, bbox = np.asarray(res["verts"])[0], np.asarray(res["pred_cams"])[0], np.asarray(res["bboxes_xyxy"])[0][:4]
    sx, sy, tx, ty = convert_crop_cam_to_orig_img(cam[None], bbox[None], w, h, 1.0, 1.25, "xyxy")[0]
    return v + np.array([tx, ty, 2 * f / (w * sx)])


# Pose (same as estimate_pose.py, method "essential")
corr = np.load(f"data/pose/{scene}/{corr_file}")
p1, p2 = corr[:, :2][:, ::-1].astype(float), corr[:, 2:][:, ::-1].astype(float)
n1 = cv2.undistortPoints(p1[:, None], K1, None)[:, 0]
n2 = cv2.undistortPoints(p2[:, None], K2, None)[:, 0]
E, mask = cv2.findEssentialMat(n1, n2, np.eye(3), cv2.RANSAC, 0.999, 7.0 / np.mean([f1, f2]), 20000)
_, R, t, _ = cv2.recoverPose(E, n1[mask.ravel() > 0], n2[mask.ravel() > 0], np.eye(3))
t = t.ravel()
m1, m2 = mesh_in_camera(folder1, f1, W1, H1), mesh_in_camera(folder2, f2, W2, H2)
def standing_height(folder):
    """Height of PARE's body for this image standing upright (same SMPL shape, rest pose), in metres. The posed
    body's vertical extent is not the person's height when they bend, crouch or raise an arm."""
    import smplx
    import torch
    if BODY == "camerahmr":
        sp = {"betas": np.load(f"data/outputs_camhmr/{folder}/body.npz")["betas"]}
    else:
        sp = np.load(f"data/outputs/{folder}/inference_result.npz", allow_pickle=True)["smpl"].item()
    rest = smplx.SMPL(model_path="data/body_models/smpl/SMPL_NEUTRAL.pkl")(
        betas=torch.tensor(sp["betas"]).float().reshape(1, 10)).vertices[0].detach().numpy()
    return float(np.ptp(rest[:, 1]))


if "height" in opts:
    # Uniform scale about each camera centre: image projections are unchanged, only the units change. The scale
    # makes the person's standing height equal the given height.
    k = float(opts["height"]) / standing_height(folder1)
    m1, m2 = m1 * k, m2 * k
    print(f"scaled scene by {k:.3f} so the person is {float(opts['height']):.2f} m tall (standing)")
s = float(t @ (m2.mean(0) - R @ m1.mean(0)))
C2 = -s * (R.T @ t)
angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
print(f"rotation {angle:.1f} deg, baseline {np.linalg.norm(C2):.2f} m")

if "ba" in opts:
    _ba = np.load(opts["ba"])
    R, _t = _ba["R_after"], _ba["t_after"]
    s = float(np.linalg.norm(_t))
    t = _t / s
    C2 = -R.T @ _t
    m1 = _ba["V_after"]            # one shared body, in camera 1's frame
    m2 = (R @ m1.T).T + _t         # the same body in camera 2's frame
    if "up_after" in _ba.files:
        # draw in the bundle adjustment's vertical: rotate camera-1 coordinates so its 'up' becomes -y
        _u = _ba["up_after"] / np.linalg.norm(_ba["up_after"])
        _ax = np.cross(_u, [0, -1.0, 0])
        if np.linalg.norm(_ax) > 1e-9:
            ALIGN = cv2.Rodrigues((_ax / np.linalg.norm(_ax) * np.arccos(np.clip(-_u[1], -1, 1))).reshape(3, 1))[0]
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    print(f"bundle-adjusted pose: rotation {angle:.1f} deg, baseline {np.linalg.norm(C2):.2f} m")

if "cam1_height" in opts:
    _A = globals().get("ALIGN", np.eye(3))
    _up = _A.T @ np.array([0, -1.0, 0])                  # vertical in image-1 camera coordinates
    _floor = (m1 @ _up).min()                             # the person's feet
    _cam1 = C2 if SWAP else np.zeros(3)                   # camera 1 as named on the page
    _k2 = float(opts["cam1_height"]) / float(_cam1 @ _up - _floor)
    m1, m2, C2, s = m1 * _k2, m2 * _k2, C2 * _k2, s * _k2
    print(f"rescaled scene by {_k2:.3f} so camera 1 is {float(opts['cam1_height']):.3f} m high; "
          f"person is now {np.ptp(m1 @ _up) * _k2 / _k2:.2f} m from feet to highest point")

# World frame for display: camera 1 frame with y flipped (up is up), z = depth from camera 1
FLIP = np.array([1.0, -1.0, 1.0])
ALIGN = globals().get("ALIGN", np.eye(3))  # rotation into the frame whose 'up' is -y (identity: camera 1 level)
to_plot = lambda P: (np.asarray(P) @ ALIGN.T) * FLIP  # noqa: E731
body = to_plot(m1)
floor_y = body[:, 1].min()  # feet
FT = 3.28084  # metres -> feet for labels


def frustum(Rc, C, K, w, h, depth):
    """Apex and the four image-plane corners (TL, TR, BR, BL) in camera-1 coordinates."""
    corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    rays = (np.linalg.inv(K) @ np.c_[corners, np.ones(4)].T).T * depth
    return C, (Rc.T @ rays.T).T + C


def draw(theme):
    light = theme == "light"
    bg = "#f2efe8" if light else "#161513"
    ink = "#0b0b0b" if light else "#f2f0ea"
    grid = "#d6d2c8" if light else "#3a3833"
    body_col = "#8f8d87" if light else "#a19f98"
    cams = [(NAMES[0], cam_colour(0, light), np.eye(3), np.zeros(3), K1, W1, H1, img1),
            (NAMES[1], cam_colour(1, light), R, C2, K2, W2, H2, img2)]

    fig = plt.figure(figsize=(10, 5.6), dpi=220)
    fig.patch.set_facecolor(bg)
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_facecolor(bg)

    # frustum depth ~1/3 of camera 1's distance to the person (1.6 m at the original 4.7 m)
    depth = 0.34 * np.linalg.norm(body.mean(0))
    centres = []
    # floor grid under the scene only (0.5 m cells, bounded to the content)
    allx = np.r_[body[:, 0], 0.0, (C2 * FLIP)[0]]
    allz = np.r_[body[:, 2], 0.0, (C2 * FLIP)[2]]
    xs = np.arange(np.floor(allx.min()) - 1.0, np.ceil(allx.max()) + 1.01, 0.5)
    zs = np.arange(np.floor(allz.min()) - 1.0, np.ceil(allz.max()) + 1.01, 0.5)
    for x in xs:
        ax.plot([x, x], [zs[0], zs[-1]], [floor_y, floor_y], color=grid, lw=0.5, zorder=0)
    for z in zs:
        ax.plot([xs[0], xs[-1]], [z, z], [floor_y, floor_y], color=grid, lw=0.5, zorder=0)

    # body (BODY_DRAW_SCALE > 1 would draw it taller than true scale, about its feet)
    BODY_DRAW_SCALE = 1.0
    foot = np.array([body[:, 0].mean(), floor_y, body[:, 2].mean()])
    b = (foot + BODY_DRAW_SCALE * (body - foot))[::3]
    ax.scatter(b[:, 0], b[:, 2], b[:, 1], s=0.35, c=body_col, depthshade=False, zorder=2, linewidths=0)

    for label, col, Rc, C, K, w, h, img in cams:
        apex, cs = frustum(Rc, C, K, w, h, depth)
        apex, cs = to_plot(apex), to_plot(cs)
        # photo on the image plane: bilinear patch over the four corners
        small = cv2.resize(img, (110, int(110 * h / w)), interpolation=cv2.INTER_AREA) / 255.0
        gh, gw = small.shape[:2]
        u = np.linspace(0, 1, gw + 1)[None, :]
        v = np.linspace(0, 1, gh + 1)[:, None]
        tl, tr, br, bl = cs
        P = ((1 - u) * (1 - v))[..., None] * tl + (u * (1 - v))[..., None] * tr + (u * v)[..., None] * br + ((1 - u) * v)[..., None] * bl
        ax.plot_surface(P[..., 0], P[..., 2], P[..., 1], facecolors=small, rstride=1, cstride=1,
                        shade=False, linewidth=0, antialiased=False, zorder=3)
        for k in range(4):
            ax.plot([apex[0], cs[k, 0]], [apex[2], cs[k, 2]], [apex[1], cs[k, 1]], color=col, lw=1.4, zorder=2.5)
            a, c = cs[k], cs[(k + 1) % 4]
            ax.plot([a[0], c[0]], [a[2], c[2]], [a[1], c[1]], color=col, lw=1.8, zorder=4)
        ax.scatter([apex[0]], [apex[2]], [apex[1]], color=col, s=28, zorder=5, depthshade=False)
        # label below the camera, on the floor-facing side
        ax.text(apex[0], apex[2], floor_y, f"{label}\n{(apex[1] - floor_y) * FT:.1f} ft high", color=ink,
                fontsize=10, ha="center", va="top", zorder=6)
        ax.plot([apex[0], apex[0]], [apex[2], apex[2]], [floor_y, apex[1]], color=col, lw=0.8, ls=(0, (2, 2)), zorder=4)
        centres.append(apex)

    ax.set_xlim(xs[0], xs[-1])
    ax.set_ylim(zs[0], zs[-1])
    zlo, zhi = floor_y, max(body[:, 1].max(), max(c[1] for c in centres)) + 0.6
    ax.set_zlim(zlo, zhi)
    ax.set_box_aspect((xs[-1] - xs[0], zs[-1] - zs[0], zhi - zlo), zoom=1.35)
    ax.view_init(elev=22, azim=-62)
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    path = f"{out_prefix}_{theme}.png"
    fig.savefig(path, facecolor=bg, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    # crop the empty margin the 3D axes leave around the scene
    im = cv2.imread(path)
    bgc = np.array([int(bg[i:i + 2], 16) for i in (5, 3, 1)])  # BGR
    ys, xs_ = np.nonzero((np.abs(im.astype(int) - bgc).sum(2) > 12))
    pad = 30
    im = im[max(0, ys.min() - pad):ys.max() + pad, max(0, xs_.min() - pad):xs_.max() + pad]
    cv2.imwrite(path, im)


def hex_bgr(h):
    return tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))


def draw_eye(theme, view_angle=65.0, eye_height_ft=4.0, eye_dist_m=None, W=2400, H=1400):
    """Perspective view from a virtual eye: eye_height_ft above the floor, swung view_angle degrees around
    the person from camera 1 towards camera 2, looking at the person. Rendered with a pinhole projection:
    floor grid, body points, frustums with each photo warped onto its image plane."""
    light = theme == "light"
    bg = "#f2efe8" if light else "#161513"
    ink = "#0b0b0b" if light else "#f2f0ea"
    grid = "#d6d2c8" if light else "#3a3833"
    body_col = "#8f8d87" if light else "#a19f98"
    cams = [("Camera 1", "#2a78d6" if light else "#3987e5", np.eye(3), np.zeros(3), K1, W1, H1, img1),
            ("Camera 2", "#eb6834" if light else "#d95926", R, C2, K2, W2, H2, img2)]
    up = np.array([0.0, 1.0, 0.0])

    # eye position: around the person's centre on the floor plane
    P = body.mean(0)
    flat = lambda v: np.array([v[0], 0.0, v[2]])  # noqa: E731
    u1 = flat(to_plot(np.zeros(3)) - P); u1 /= np.linalg.norm(u1)
    u2 = flat(to_plot(C2) - P); u2 /= np.linalg.norm(u2)
    sgn = np.sign(np.cross(u1, u2)[1]) or 1.0  # rotate from camera 1 towards camera 2
    a = np.radians(view_angle) * sgn
    d = np.array([u1[0] * np.cos(a) + u1[2] * np.sin(a), 0.0, -u1[0] * np.sin(a) + u1[2] * np.cos(a)])
    if np.dot(d, u2) < np.dot(u1, u2):  # wrong turning direction: flip
        a = -a
        d = np.array([u1[0] * np.cos(a) + u1[2] * np.sin(a), 0.0, -u1[0] * np.sin(a) + u1[2] * np.cos(a)])
    dist = eye_dist_m or 1.5 * max(np.linalg.norm(flat(to_plot(np.zeros(3)) - P)), np.linalg.norm(flat(to_plot(C2) - P)))
    eye = np.array([P[0], floor_y + eye_height_ft / FT, P[2]]) + dist * d
    target = np.array([P[0], floor_y + 0.5 * np.ptp(body[:, 1]), P[2]])
    fwd = target - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    vup = np.cross(right, fwd)

    def cam_xyz(X):
        r = np.atleast_2d(X) - eye
        return np.c_[r @ right, r @ vup, r @ fwd]

    # focal length chosen so every camera centre, frustum and the body fit with a margin
    depth = float(np.clip(0.3 * np.linalg.norm(body.mean(0)), 0.5, 1.6))
    geo = [frustum(Rc, C, K, w, h, depth) for _, _, Rc, C, K, w, h, _ in cams]
    key = np.vstack([body] + [np.vstack([to_plot(ap)[None], to_plot(cs)]) for ap, cs in geo]
                    + [np.array([[to_plot(ap)[0], floor_y, to_plot(ap)[2]]]) for ap, _ in geo])
    c = cam_xyz(key)
    nx, ny = c[:, 0] / c[:, 2], -c[:, 1] / c[:, 2]
    f = 0.88 * min(W / np.ptp(nx), H / np.ptp(ny))
    off = np.array([W / 2 - f * (nx.max() + nx.min()) / 2, H / 2 - f * (ny.max() + ny.min()) / 2])

    def proj(X):
        q = cam_xyz(X)
        return np.c_[f * q[:, 0] / q[:, 2], -f * q[:, 1] / q[:, 2]] + off, q[:, 2]

    SS = 2  # supersample for smooth lines
    canvas = np.full((H * SS, W * SS, 3), hex_bgr(bg), np.uint8)

    def line(Pa, Pb, colour, w, n=1, dashed=False):
        pts = np.linspace(Pa, Pb, max(2, n))
        uv, z = proj(pts)
        for k in range(len(pts) - 1):
            if z[k] <= 0.2 or z[k + 1] <= 0.2 or (dashed and k % 2):
                continue
            cv2.line(canvas, tuple(np.round(uv[k] * SS).astype(int)), tuple(np.round(uv[k + 1] * SS).astype(int)),
                     hex_bgr(colour), int(w * SS), cv2.LINE_AA)

    # floor grid (0.5 m cells around the scene)
    allx = np.r_[body[:, 0], 0.0, to_plot(C2)[0]]
    allz = np.r_[body[:, 2], 0.0, to_plot(C2)[2]]
    gx = np.arange(np.floor(allx.min()) - 1.0, np.ceil(allx.max()) + 1.01, 0.5)
    gz = np.arange(np.floor(allz.min()) - 1.0, np.ceil(allz.max()) + 1.01, 0.5)
    for x in gx:
        line(np.array([x, floor_y, gz[0]]), np.array([x, floor_y, gz[-1]]), grid, 1.6, n=60)
    for z in gz:
        line(np.array([gx[0], floor_y, z]), np.array([gx[-1], floor_y, z]), grid, 1.6, n=60)

    # objects back to front
    items = [("body", None)] + [("cam", i) for i in range(2)]
    dist_of = lambda it: (np.linalg.norm(body.mean(0) - eye) if it[0] == "body"  # noqa: E731
                          else np.linalg.norm(to_plot(geo[it[1]][1]).mean(0) - eye))
    labels = []
    for kind, i in sorted(items, key=dist_of, reverse=True):
        if kind == "body":
            uv, z = proj(body)
            order = np.argsort(-z)
            for (x, y) in uv[order]:
                cv2.circle(canvas, (int(x * SS), int(y * SS)), int(2.2 * SS), hex_bgr(body_col), -1, cv2.LINE_AA)
            continue
        label, col, _, _, _, w, h, img = cams[i]
        apex, cs = to_plot(geo[i][0]), to_plot(geo[i][1])
        foot = np.array([apex[0], floor_y, apex[2]])
        line(foot, apex, col, 2.0, n=40, dashed=True)
        # photo warped onto the image plane
        uv, z = proj(cs)
        if (z > 0.2).all():
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            M = cv2.getPerspectiveTransform(src, np.float32(uv * SS))
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            warped = cv2.warpPerspective(bgr, M, (W * SS, H * SS), flags=cv2.INTER_AREA)
            m = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W * SS, H * SS)) > 127
            canvas[m] = warped[m]
        for k in range(4):
            line(apex, cs[k], col, 3.4)
            line(cs[k], cs[(k + 1) % 4], col, 4.4)
        a_uv, _ = proj(apex)
        cv2.circle(canvas, tuple(np.round(a_uv[0] * SS).astype(int)), int(11 * SS), hex_bgr(col), -1, cv2.LINE_AA)
        f_uv, _ = proj(foot)
        labels.append((f_uv[0], f"{label}\n{(apex[1] - floor_y) * FT:.1f} ft high"))

    canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_AREA)
    fig = plt.figure(figsize=(W / 220, H / 220), dpi=220)
    fig.patch.set_facecolor(bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    for (x, y), txt in labels:
        ax.text(x, y + 14, txt, color=ink, fontsize=15, ha="center", va="top")
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.set_axis_off()
    path = f"{out_prefix}_{theme}.png"
    fig.savefig(path, facecolor=bg, dpi=220)
    plt.close(fig)
    im = cv2.imread(path)
    bgc = np.array(hex_bgr(bg))
    yy, xx = np.nonzero((np.abs(im.astype(int) - bgc).sum(2) > 12))
    pad = 30
    cv2.imwrite(path, im[max(0, yy.min() - pad):yy.max() + pad, max(0, xx.min() - pad):xx.max() + pad])


def draw_top(theme):
    """Bird's-eye view: camera wedges, a line from each camera to the person with its length, and the angle
    between those lines at the person. Rotated so camera 1 faces the top of the page."""
    # (x, z) on the floor -> page coords, rotated so camera 1 (as named on the page) faces the top of the page
    _c1R = R if SWAP else np.eye(3)
    _d = ALIGN @ (_c1R.T @ np.array([0, 0, 1.0]))
    _phi = np.pi / 2 - np.arctan2(-_d[0], _d[2])
    _rot = np.array([[np.cos(_phi), -np.sin(_phi)], [np.sin(_phi), np.cos(_phi)]])
    top = lambda x, z: _rot @ np.array([z, -x])  # noqa: E731
    from matplotlib.patches import Arc, Polygon
    light = theme == "light"
    bg = "#f2efe8" if light else "#161513"
    ink = "#0b0b0b" if light else "#f2f0ea"
    muted = "#6b6862" if light else "#9a978f"
    body_col = "#8f8d87" if light else "#a19f98"
    cols = [cam_colour(0, light), cam_colour(1, light)]
    cams = [(np.eye(3), np.zeros(3), K1, W1, NAMES[0]), (R, C2, K2, W2, NAMES[1])]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=220)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    bxy = top(body[:, 0], body[:, 2]).T
    ax.scatter(bxy[::2, 0], bxy[::2, 1], s=0.5, c=body_col, linewidths=0, zorder=2)

    L = 1.3
    dirs, centres, label_pts = [], [], []
    for (Rc, C, K, w, label), col in zip(cams, cols):
        d = ALIGN @ (Rc.T @ np.array([0, 0, 1.0]))
        half = np.arctan(w / 2 / K[0, 0])
        d2 = top(d[0], d[2]); d2 /= np.linalg.norm(d2)
        rot = lambda a: np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])  # noqa: E731
        Ca = ALIGN @ C
        c2 = top(Ca[0], Ca[2])
        wedge = [c2, c2 + L * rot(half) @ d2, c2 + L * rot(-half) @ d2]
        ax.add_patch(Polygon(wedge, closed=True, fc=col, ec=col, alpha=0.18, lw=0, zorder=3))
        ax.add_patch(Polygon(wedge, closed=True, fill=False, ec=col, lw=1.6, joinstyle="round", zorder=4))
        ax.scatter(*c2, color=col, s=30, zorder=5)
        lab_pos = c2 - 0.55 * d2  # behind the camera, away from its wedge
        ax.text(*lab_pos, label, color=ink, fontsize=10, ha="center", va="center", zorder=6)
        label_pts.append(lab_pos)
        dirs.append(d2); centres.append(c2)

    # a line from each camera to the person's centre, labelled with its length, and the angle between the
    # two lines at the person (the angle around the person from one camera to the other)
    bc = bxy.mean(0)
    for k, (c2, col) in enumerate(zip(centres, cols)):
        ax.plot([c2[0], bc[0]], [c2[1], bc[1]], color=col, lw=1.0, ls=(0, (3, 3)), zorder=3)
        v = bc - c2
        nrm = np.array([-v[1], v[0]]) / np.linalg.norm(v)
        other = centres[1 - k]
        nrm = -nrm if nrm @ (other - (c2 + 0.5 * v)) > 0 else nrm  # label on the side away from the other camera
        mp = c2 + 0.5 * v + 0.25 * nrm
        ang_txt = np.degrees(np.arctan2(v[1], v[0]))
        ang_txt = ang_txt - 180 if ang_txt > 90 else ang_txt + 180 if ang_txt < -90 else ang_txt
        ax.text(*mp, f"{np.linalg.norm(v) * FT:.1f} ft", color=muted, fontsize=9, ha="center", va="center",
                rotation=ang_txt, rotation_mode="anchor", zorder=6)
    u = [(c2 - bc) / np.linalg.norm(c2 - bc) for c2 in centres]
    ang = np.degrees(np.arccos(np.clip(u[0] @ u[1], -1, 1)))
    th = [np.degrees(np.arctan2(v[1], v[0])) for v in u]
    t0, t1 = sorted(th)
    if t1 - t0 > 180:
        t0, t1 = t1, t0 + 360
    ax.add_patch(Arc(bc, 1.3, 1.3, theta1=t0, theta2=t1, color=ink, lw=1.0, zorder=5))
    mid = np.radians((t0 + t1) / 2)
    ax.text(bc[0] + 1.0 * np.cos(mid), bc[1] + 1.0 * np.sin(mid), f"{ang:.0f}°", color=ink, fontsize=11,
            ha="center", va="center", zorder=6)
    print(f"angle around the person between the cameras (top-down) {ang:.1f} deg")

    pts = np.vstack(centres + label_pts + [bxy])
    lo, hi = pts.min(0) - 1.0, pts.max(0) + 1.0

    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_aspect("equal"); ax.set_axis_off()
    path = f"{out_prefix}_top_{theme}.png"
    fig.savefig(path, facecolor=bg, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def draw_projection(mask_path):
    """Camera 2's PARE mesh moved into camera 1 with the recovered pose and projected into image 1 (red),
    over the person's DensePose mask in image 1 (blue). Blue showing around the red means a miss."""
    import torch
    faces = np.load("data/body_models/smpl_faces.npy").astype(int)
    X1 = (R.T @ (m2 - s * t).T).T
    q = (K1 @ X1.T).T
    uv = q[:, :2] / q[:, 2:3]

    dp = torch.load(open(mask_path, "rb"))[0]
    box, lab = dp["pred_boxes_XYXY"][0].numpy(), dp["pred_densepose"][0].labels.cpu().numpy()
    mask = np.zeros((H1, W1), bool)
    x0, y0 = int(box[0]), int(box[1])
    hh, ww = min(lab.shape[0], H1 - y0), min(lab.shape[1], W1 - x0)
    mask[y0:y0 + hh, x0:x0 + ww] = lab[:hh, :ww] > 0

    base = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
    blue = base.copy()
    blue[mask] = (235, 90, 30)
    out = cv2.addWeighted(base, 0.35, blue, 0.65, 0)
    out[~mask] = base[~mask]
    red = np.zeros((H1, W1), np.uint8)
    for f in faces[np.argsort(-X1[faces].mean(1)[:, 2])]:
        cv2.fillPoly(red, [np.round(uv[f]).astype(np.int32)], 255, cv2.LINE_AA)
    a = (red.astype(float) / 255 * 0.85)[..., None]
    out = (out * (1 - a) + np.array([40, 40, 225]) * a).astype(np.uint8)
    cv2.imwrite(f"{out_prefix}_projection.jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 88])


if len(sys.argv) > 7:
    draw_projection(sys.argv[7])
for theme in ("light", "dark"):
    draw(theme)
    draw_top(theme)
print("wrote", f"{out_prefix}_light.png", f"{out_prefix}_dark.png", f"{out_prefix}_top_light.png", f"{out_prefix}_top_dark.png")

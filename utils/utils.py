from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import yaml
from mmhuman3d.core.cameras.cameras import WeakPerspectiveCameras
from tqdm import tqdm

import utils.densepose_utils as dp_utils
from utils.visualization import visualize_smpl_vibe


def get_smpl_mesh(smpl_path: str, dims: Tuple[int, int] = None):
    """Parses SMPL output and returns predicted cameras, vertices, and faces.

    Keys from inference_results.npz for reference: ['__key_strict__', '__data_len__', '__keypoints_compressed__', 'smpl', 'verts', 'pred_cams', 'bboxes_xyxy', 'image_path', 'person_id', 'frame_id'].

    Args:
        smpl_path: Path to inference_results.npz output by estimate_smpl.py.
        dims: Tuple containing (H, W) of image.
    """

    output = np.load(smpl_path, allow_pickle=True)
    smpl_params = output["smpl"]
    vertices = output["verts"]
    faces = np.load("./data/body_models/smpl_faces.npy")
    cams = output["pred_cams"]
    bbox = output["bboxes_xyxy"]
    assert dims is not None

    body_config = {"model_path": "./data/body_models/", "type": "smpl"}
    cameras, vertices_new = visualize_smpl_vibe(
        pred_cam=cams,
        bbox=bbox,
        output_path=str(Path(smpl_path).parent),
        resolution=dims,
        verts=vertices,
        body_model_config=body_config,
        overwrite=True,
    )
    return cameras, vertices_new, faces


def project_points(
    cameras: WeakPerspectiveCameras,
    vertices: torch.FloatTensor,
    dims: Tuple[int, int],
    img2_coords: np.ndarray = None,
):
    """Projects mesh vertices using input camera and checks bounds of correspondences.

    Args:
        cameras: WeakPerspectiveCameras returned from visualize_smpl_vibe().
        vertices: Predicted SMPL vertices in world coordinates.
        dims: Tuple containing (H, W) of image.
        img2_coords: When the vertices argument in this function correspond to the densepose vertices returned from parse_densepose_data(), this argument should contain the pixel locations corresponding to the vertices to ensure that any points projected to image 1 that get filtered out because they are out of bounds, are also filtered out in the list of pixel locations in image 2.
    """
    # Handle numpy arrays.
    if isinstance(vertices, np.ndarray):
        vertices = torch.from_numpy(vertices).reshape((1, -1, 3))

    # Output should be (N, 3) list of NDC coordinates.
    ndc_coords = torch.squeeze(
        cameras.get_full_projection_transform().transform_points(
            vertices.to(cameras.device)
        )
    )

    # Edge case where there is only one vertex.
    if ndc_coords.dim() <= 1:
        ndc_coords = ndc_coords.reshape(-1, 3)

    # Following https://pytorch3d.org/docs/cameras#camera-coordinate-systems, determine dimensions of NDC object volume and appropriately transform to screen-coordinates.
    H, W = dims
    dims = (W, H)
    if H > W:
        s = H / W
        short_idx = 0
        long_idx = 1
    else:
        s = W / H
        short_idx = 1
        long_idx = 0

    ndc_coords[:, short_idx] = (ndc_coords[:, short_idx] + 1.0) / 2.0
    ndc_coords[:, short_idx] *= dims[short_idx]
    ndc_coords[:, short_idx] = dims[short_idx] - ndc_coords[:, short_idx]

    ndc_coords[:, long_idx] = (ndc_coords[:, long_idx] + s) / (2 * s)
    ndc_coords[:, long_idx] *= dims[long_idx]
    ndc_coords[:, long_idx] = dims[long_idx] - ndc_coords[:, long_idx]

    # Filter points to ensure all projected points are within bounds of image.
    img1_coords = ndc_coords[:, :-1].to(int).cpu().numpy()
    vertices = torch.squeeze(vertices).cpu().numpy()
    filter_bounds = lambda coords, arr: arr[coords]
    filtered_coords = np.where(
        (img1_coords[:, 1] > 0)
        & (img1_coords[:, 1] < H)
        & (img1_coords[:, 0] > 0)
        & (img1_coords[:, 0] < W)
    )
    img1_coords = filter_bounds(filtered_coords, img1_coords)
    vertices = filter_bounds(filtered_coords, vertices)
    if img2_coords is not None:
        img2_coords = filter_bounds(filtered_coords, img2_coords)

    if img2_coords is not None:
        return vertices, img1_coords[:, ::-1], img2_coords
    else:
        return vertices, img1_coords[:, ::-1]


def parse_densepose_data(
    densepose_path: str,
    smpl_vertices: np.ndarray,
    visualize: bool = False,
    get_xyz: bool = True,
    person_id: int = 0,
    part_idx: int = None,
):
    """Parses DensePose output and returns IUV or XYZ coordinates.

    Args:
        densepose_path: Path to densepose.pt output from estimate_densepose.py.
        smpl_vertices: SMPL vertices from get_smpl_mesh() for calculating IUV -> XYZ transform.
        visualize: Flag to visualize IUV coordinates.
        get_xyz: Flag to determine whether to return IUV coordinates or XYZ coordinates.
        person_id: Important for multi-person scenes.
        part_idx: Optionally passed to iuv_to_xyz() to filter densepose vertices and pixel locations by part (mainly for visualization purposes).

    """
    f = open(densepose_path, "rb")
    data = torch.load(f)[0]

    # Object detection confidences.
    scores = data["scores"][person_id]
    # Predicted bounding boxes.
    bboxes = data["pred_boxes_XYXY"][person_id]  # Top Left Bottom Right
    # DensePoseChartResultWithConfidences.
    densepose = data["pred_densepose"][person_id]
    # H x W array of chart indices where H, W are the height, width of the bounding box and chart indices denote which body part the pixel belongs to.
    labels = densepose.labels.cpu().numpy()
    # 2 x H x W array of uv coordinates. Same H, W as above.
    uv = densepose.uv.cpu().numpy()

    if visualize:
        mask = np.zeros_like(labels)
        mask[np.where(labels)] = 255
        cv2.imwrite(str(Path(densepose_path).parent / f"mask_{person_id}.png"), mask)

        cv2.imwrite(
            str(Path(densepose_path).parent / f"I_{person_id}.png"),
            ((labels / 24.0) * 255).astype(np.uint8),
        )

        cv2.imwrite(
            str(Path(densepose_path).parent / f"U_{person_id}.png"),
            ((uv[0]) * 255).astype(np.uint8),
        )

        cv2.imwrite(
            str(Path(densepose_path).parent / f"V_{person_id}.png"),
            ((uv[1]) * 255).astype(np.uint8),
        )

    IUV = np.dstack([labels, uv[0], uv[1]]).astype(np.float32)
    if get_xyz:
        XYZ, pixel_locations = iuv_to_xyz(
            IUV, smpl_vertices=smpl_vertices, part_idx=part_idx
        )
        pixel_locations += np.array([bboxes[1], bboxes[0]]).astype(
            pixel_locations.dtype
        )
        return XYZ, pixel_locations

    return IUV


def iuv_to_xyz(IUV: np.ndarray, smpl_vertices: np.ndarray, part_idx: int = None):
    """Function from https://github.com/davidleejy/DensePose/blob/speedup/notebooks/DensePose-Fast-IUV-2-XYZ.ipynb for projecting IUV DensePose coordinates to XYZ SMPL coordinates.

    Args:
        IUV: IUV coordinates from parse_densepose_data().
        smpl_vertices: SMPL vertices from get_smpl_mesh().
        part_idx: Optional part index to filter by part (mainly for visualization purposes).
    """
    X, Y, Z = smpl_vertices[:, 0], smpl_vertices[:, 1], smpl_vertices[:, 2]

    DP = dp_utils.DensePoseMethods(Path("./data"))
    # Mask for pixels belonging to person and optionally part.
    pixel_locations = (
        np.where(IUV[..., 0] == part_idx)
        if part_idx is not None
        else np.where(IUV[..., 0])
    )
    points_iuv = IUV[pixel_locations]

    points_xyz = np.zeros_like(points_iuv)
    for i, point_iuv in tqdm(enumerate(points_iuv)):
        FaceIndex, bc1, bc2, bc3 = DP.IUV2FBC_fast(
            point_iuv[0], point_iuv[1], point_iuv[2]
        )
        point_xyz = DP.FBC2PointOnSurface(FaceIndex, bc1, bc2, bc3, smpl_vertices)
        points_xyz[i] = point_xyz

    pixel_locations = np.hstack(
        [pixel_locations[0].reshape(-1, 1), pixel_locations[1].reshape(-1, 1)]
    )
    return points_xyz, pixel_locations


def parse_config(config_path: str):
    """Loads config dictionary from config.yaml file.

    Args:
        config_path: Path to config.yaml file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        f.close()

    return config

import argparse as ap
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

import utils.utils as utils


def predict_densepose(image_path: Path, config: dict):
    """Runs densepose on given image and gets IUV and corresponding XYZ coordinates for each human in scene.

    Args:
        image_path: Path to image to run inference on.
        config: Dictionary containing pipeline settings.
    """
    densepose_config = config["densepose_config"]
    densepose_checkpoint = config["densepose_checkpoint"]
    out_dir = Path(config["out_dir"])
    out_path = f"{out_dir / f'outputs/{image_path.parent.stem}/densepose.pt'}"
    cmd = [
        "python",
        "estimate_densepose.py",
        "dump",
        f"{densepose_config}",
        f"{densepose_checkpoint}",
        f"{image_path}",
        "--output",
        f"{out_path}",
        "-v",
    ]

    subprocess.run(cmd)

    return out_path


def predict_smpl(image_path: Path, config: dict):
    """Predicts SMPL and camera parameters for given image.

    Args:
        image_path: Path to image to run inference on.
        config: Dictionary containing pipeline settings.
    """
    smpl_model_config = config["smpl_model_config"]
    smpl_model_checkpoint = config["smpl_model_checkpoint"]
    det_model_config = config["detection_model_config"]
    det_model_checkpoint = config["detection_model_checkpoint"]
    tracking_model_config = config["tracking_model_config"]

    out_dir = Path(config["out_dir"])
    out_dir = f"{out_dir / f'outputs/{image_path.parent.stem}'}"
    if config["multi_person"]:
        cmd = [
            "python",
            "estimate_smpl.py",
            f"{smpl_model_config}",
            f"{smpl_model_checkpoint}",
            "--tracking_config",
            f"{tracking_model_config}",
            "--input_path",
            f"{image_path}",
            "--show_path",
            out_dir,
            "--output",
            out_dir,
            "--multi_person_demo",
        ]
    else:
        cmd = [
            "python",
            "estimate_smpl.py",
            f"{smpl_model_config}",
            f"{smpl_model_checkpoint}",
            "--det_config",
            f"{det_model_config}",
            "--det_checkpoint",
            f"{det_model_checkpoint}",
            "--input_path",
            f"{image_path}",
            "--show_path",
            out_dir,
            "--output",
            out_dir,
            "--single_person_demo",
        ]

    subprocess.run(cmd)

    input_image = cv2.imread(str(image_path))
    H, W, _ = input_image.shape
    cameras, vertices, faces = utils.get_smpl_mesh(
        Path(out_dir) / "inference_result.npz", dims=(H, W)
    )

    return cameras, vertices, faces


def get_virtual_correspondences(
    image1_path: Path,
    image2_path: Path,
    config: dict,
    part_idx: int = None,
    save_correspondences=True,
    visualize=True,
    frame_idx=1,
):
    """Computes virtual correspondences between 2 images.

    Args:
        image1_path: Path to image that will be passed into human reconstruction model.
        image2_path: Path to image that will be passed into DensePose.
        config: Dictionary containing pipeline settings.
        part_idx: Optionally include part index to filter predictions and visualizations by part.
        save_correspondences = Flag to save out correspondences.
        visualize = Flag to visualize correspondences.

    TODO (nwijayaratne): ReID net for multi-person tracking
    """
    cameras, vertices, faces = predict_smpl(image1_path, config)
    cameras2, vertices2, faces2 = predict_smpl(image2_path, config)
    # vertices2[0] -= cameras.T.to(vertices2[0].device) - cameras2.T.to(vertices2[0].device)
    # vertices2[0] = (torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=cameras.R.dtype, device=vertices2[0].device) @ vertices2[0].T).T
    # vertices2[0] = cameras.R.T.to(vertices2[0].device) @ vertices2[0]
    # vertices2[0] = (torch.tensor([[ 0.51128958, -0.01874729, -0.859204] , [-0.8593316, 0.00221703, -0.51141392], [0.01149251,  0.9998218,  -0.01497659]], dtype=cameras.R.dtype, device=vertices2[0].device) @ vertices2[0].T).T
    # print(torch.tensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=cameras.R.dtype, device=vertices2[0].device))
    # print(torch.squeeze(cameras2.R).T @ torch.squeeze(cameras.R))
    # vertices2[0][:, 0] = vertices[0][:, 0] - (cameras.T[0][-1] - cameras2.T[0][-1]).to(vertices[0].device)

    densepose_path = predict_densepose(image2_path, config)

    img1 = cv2.imread(str(image1_path))
    img2 = cv2.imread(str(image2_path))

    img2_padding = 0
    img1_padding = 0
    # Add vertical padding for valid horizontal stack.
    if img1.shape[0] > img2.shape[0]:
        padding = (img1.shape[0] - img2.shape[0]) // 2
        if padding < 1:
            padding = 1
            img2 = np.pad(img2, ((padding, 0), (0, 0), (0, 0)))
        else:
            if (img1.shape[0] - img2.shape[0]) % 2 != 0:
                img2 = np.pad(img2, ((padding, padding + 1), (0, 0), (0, 0)))
            else:
                img2 = np.pad(img2, ((padding, padding), (0, 0), (0, 0)))
        img2_padding += padding
    elif img2.shape[0] > img1.shape[0]:
        padding = (img2.shape[0] - img1.shape[0]) // 2
        if padding < 1:
            padding = 1
            img1 = np.pad(img1, ((padding, 0), (0, 0), (0, 0)))
        else:
            if (img2.shape[0] - img1.shape[0]) % 2 != 0:
                img1 = np.pad(img1, ((padding, padding + 1), (0, 0), (0, 0)))
            else:
                img1 = np.pad(img1, ((padding, padding), (0, 0), (0, 0)))
        img1_padding += padding

    img_stack = np.hstack([img1, img2])
    img_stack2 = np.hstack([img1, img2])
    colors = [
        (255, 0, 0),
        (0, 0, 255),
    ]  # TODO (niviruwijayaratne): Maybe don't hardcode this

    correspondences = None
    vertices2 = vertices
    for person_id in tqdm(range(min(len(vertices), len(vertices2)))):
        # For specific person, get DensePose_vertices in world space and corresponding pixel coordinates in the second image.
        densepose_vertices, pixel_locations = utils.parse_densepose_data(
            densepose_path,
            np.squeeze(vertices[person_id]),
            get_xyz=True,
            visualize=True,
            person_id=person_id,
            part_idx=part_idx,
        )
        # Project and filter SMPL vertices to image 1.
        smpl_vertices, smpl_projections = utils.project_points(
            cameras[person_id], vertices[person_id], dims=(img1.shape[0], img1.shape[1])
        )
        densepose_vertices2, densepose_projectiosn2 = utils.project_points(
            cameras[person_id], densepose_vertices, dims=(img1.shape[0], img1.shape[1])
        )
        canvas = np.zeros_like(img1)
        canvas[smpl_projections[:, 0], smpl_projections[:, 1]] = [255, 0, 0]
        canvas[densepose_projectiosn2[:, 0], densepose_projectiosn2[:, 1]] = [0, 0, 255]
        cv2.imwrite("./data/outputs/messi/densepose_mapping.png", canvas)

        # Project mesh from image 2 onto image 1
        smpl_vertices2, smpl_projections2 = utils.project_points(
            cameras[person_id],
            vertices2[person_id],
            dims=(img1.shape[0], img1.shape[1]),
        )

        (
            densepose_vertices,
            img1_coords,
            img2_coords,
        ) = utils.project_points(
            cameras[person_id],
            densepose_vertices,
            dims=(img1.shape[0], img1.shape[1]),
            img2_coords=pixel_locations,
        )
        if correspondences is None:
            correspondences = np.hstack([img1_coords, img2_coords])
        else:
            correspondences = np.vstack(
                [correspondences, np.hstack([img1_coords, img2_coords])]
            )
        """Visualization"""
        if visualize:
            # Adjust projected image coordinates based on padding.
            img1_coords[:, 0] += img1_padding
            img2_coords[:, 0] += img2_padding

            # Find skip value to visualize 100 correspondences.
            if len(img1_coords) // 100 == 0:
                skip = 1
            else:
                skip = len(img1_coords) // 50

            img1_coords = img1_coords[::skip]
            img2_coords = img2_coords[::skip]
            for img1_coord, img2_coord in zip(smpl_projections, smpl_projections2):
                cv2.circle(img_stack2, img1_coord[::-1], 3, colors[person_id], -1)
                cv2.circle(img_stack2, img2_coord[::-1], 3, colors[person_id], -1)
                cv2.line(img_stack2, img1_coord[::-1], img2_coord[::-1], (0, 255, 0), 1)

            for img1_coord, img2_coord in zip(img1_coords, img2_coords):
                img2_coord[1] += img1.shape[1]
                cv2.circle(img_stack, img1_coord[::-1], 3, colors[person_id], -1)
                cv2.circle(img_stack, img2_coord[::-1], 3, colors[person_id], -1)
                cv2.line(img_stack, img1_coord[::-1], img2_coord[::-1], (0, 255, 0), 1)

    if save_correspondences:
        out_path = (
            Path(config["out_dir"])
            / "outputs"
            / f"{image1_path.parent.stem}"
            / f"correspondences.npy"
        )
        np.save(out_path, correspondences)
    if visualize:
        if part_idx is None:
            part_idx = "all"
        out_path = (
            Path(config["out_dir"])
            / "outputs"
            / f"{image1_path.parent.stem}"
            / f"densepose_pixel_correspondence_{part_idx}_{frame_idx}.png"
        )

        out_path2 = (
            Path(config["out_dir"])
            / "outputs"
            / f"{image1_path.parent.stem}"
            / f"mesh_projections.png"
        )

        print(out_path)
        cv2.imwrite(str(out_path), img_stack)
        cv2.imwrite(str(out_path2), img_stack2)

    return correspondences


def main(args):
    config = utils.parse_config(args.config)
    if args.image1_path is not None and args.image2_path is not None:
        image1_path = Path(args.image1_path)
        image2_path = Path(args.image2_path)
        correspondences = get_virtual_correspondences(
            image1_path, image2_path, config, part_idx=args.part_idx
        )
    if args.image_dir is not None:
        ims = Path(args.image_dir).glob("*.jpg")
        ims = sorted(
            list(ims), key=lambda x: f"{int(str(Path(x.stem))):06d}", reverse=True
        )
        base_im = ims[-1]
        for i, im in enumerate(ims):
            if len(ims) - 1 - i > 100:
                continue
            else:
                correspondences = get_virtual_correspondences(
                    base_im, im, config, part_idx=args.part_idx, frame_idx=len(ims) - i
                )


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config.yaml file")
    parser.add_argument("--image1_path", type=str, help="Path to first input image.")
    parser.add_argument("--image2_path", type=str, help="Path to second input image.")
    parser.add_argument(
        "--image_dir", type=str, help="Path to directory of videp frames."
    )
    parser.add_argument("--part_idx", type=int, help="Part index", default=None)
    args = parser.parse_args()
    main(args)

#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

"""
python visualization.py --input <path to directory of .jpgs>
"""

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.cm as cm
import torch
from models.matching import Matching
from models.utils import (
    AverageTimer,
    VideoStreamer,
    frame2tensor,
    make_matching_plot_fast,
)
from tqdm import tqdm

torch.set_grad_enabled(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SuperGlue demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        # default='0',
        default="images/yuna",
        help="ID of a USB webcam, URL of an IP camera, "
        "or path to an image directory or movie file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        # default=None,
        default="outputs",
        help="Directory where to write output frames (If None, no output)",
    )

    parser.add_argument(
        "--image_glob",
        type=str,
        nargs="+",
        default=["*.png", "*.jpg", "*.jpeg"],
        help="Glob if a directory of images is specified",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Images to skip if input is a movie or directory",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1000000,
        help="Maximum length if input is a movie or directory",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs="+",
        default=[640, 480],
        help="Resize the input image before running inference. If two numbers, "
        "resize to the exact dimensions, if one number, resize the max "
        "dimension, if -1, do not resize",
    )

    parser.add_argument(
        "--superglue",
        choices={"indoor", "outdoor"},
        default="indoor",
        help="SuperGlue weights",
    )
    parser.add_argument(
        "--max_keypoints",
        type=int,
        default=-1,
        help="Maximum number of keypoints detected by Superpoint"
        " ('-1' keeps all keypoints)",
    )
    parser.add_argument(
        "--keypoint_threshold",
        type=float,
        default=0.005,
        help="SuperPoint keypoint detector confidence threshold",
    )
    parser.add_argument(
        "--nms_radius",
        type=int,
        default=4,
        help="SuperPoint Non Maximum Suppression (NMS) radius" " (Must be positive)",
    )
    parser.add_argument(
        "--sinkhorn_iterations",
        type=int,
        default=20,
        help="Number of Sinkhorn iterations performed by SuperGlue",
    )
    parser.add_argument(
        "--match_threshold", type=float, default=0.2, help="SuperGlue match threshold"
    )

    parser.add_argument(
        "--show_keypoints", action="store_true", help="Show the detected keypoints"
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        default=True,
        help="Do not display images to screen. Useful if running remotely",
    )
    parser.add_argument(
        "--force_cpu", action="store_true", help="Force pytorch to run in CPU mode."
    )

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print("Will resize to {}x{} (WxH)".format(opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print("Will resize max dimension to {}".format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print("Will not resize images")
    else:
        raise ValueError("Cannot specify more than two integers for --resize")

    device = "cuda" if torch.cuda.is_available() and not opt.force_cpu else "cpu"
    print('Running inference on device "{}"'.format(device))
    config = {
        "superpoint": {
            "nms_radius": opt.nms_radius,
            "keypoint_threshold": opt.keypoint_threshold,
            "max_keypoints": opt.max_keypoints,
        },
        "superglue": {
            "weights": opt.superglue,
            "sinkhorn_iterations": opt.sinkhorn_iterations,
            "match_threshold": opt.match_threshold,
        },
    }
    matching = Matching(config).eval().to(device)
    keys = ["keypoints", "scores", "descriptors"]

    if opt.output_dir is not None:
        print("==> Will write outputs to {}".format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    timer = AverageTimer()
    ims = sorted(
        list(Path(opt.input).glob("*.jpg")), key=lambda x: f"{int(x.stem):06d}"
    )
    base_frame_viz = cv2.imread(str(ims[0]))
    base_frame = cv2.cvtColor(base_frame_viz, cv2.COLOR_BGR2GRAY)
    frame_tensor = frame2tensor(base_frame, device)

    last_data = matching.superpoint({"image": frame_tensor})
    last_data = {k + "0": last_data[k] for k in keys}
    last_data["image0"] = frame_tensor
    last_image_id = 0

    for i, im in tqdm(enumerate(ims)):
        if i == 0:
            continue
        frame_viz = cv2.imread(str(im))
        frame = cv2.cvtColor(frame_viz, cv2.COLOR_BGR2GRAY)
        frame_tensor = frame2tensor(frame, device)
        pred = matching({**last_data, "image1": frame_tensor})
        kpts0 = last_data["keypoints0"][0].cpu().numpy()
        kpts1 = pred["keypoints1"][0].cpu().numpy()
        matches = pred["matches0"][0].cpu().numpy()
        confidence = pred["matching_scores0"][0].cpu().numpy()
        timer.update("forward")

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])
        text = []
        k_thresh = matching.superpoint.config["keypoint_threshold"]
        m_thresh = matching.superglue.config["match_threshold"]
        small_text = []
        out = make_matching_plot_fast(
            base_frame_viz,
            frame_viz,
            kpts0,
            kpts1,
            mkpts0,
            mkpts1,
            color,
            text,
            path=os.path.join(opt.output_dir, f"match_{i}.png"),
            show_keypoints=opt.show_keypoints,
            small_text=small_text,
        )

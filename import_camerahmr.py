"""Copy CameraHMR outputs (vc-modern/run_camerahmr.py) to data/outputs_camhmr/<folder>/body.npz for the pipeline.

Usage: python import_camerahmr.py <vc-modern/out_camhmr dir>
"""
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])
FOLDERS = {"apartment": ("single_view2", "single_view1"), "foot-stall": ("foot-stall-view2", "foot-stall-view1"),
           "chest-stall": ("chest-stall-view2", "chest-stall-view1"), "neck-stall": ("neck-stall-view2", "neck-stall-view1")}
for scene, folders in FOLDERS.items():
    for i, folder in enumerate(folders, 1):
        out = Path("data/outputs_camhmr") / folder
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy(src / f"{scene}_{i}.npz", out / "body.npz")
        print(scene, i, "->", out / "body.npz")

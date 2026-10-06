"""Convert the official SMPL v1.1.0 neutral model to a Python 3 pickle without chumpy.

Usage: python convert_smpl.py <basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl>
Writes data/body_models/smpl/SMPL_NEUTRAL.pkl and data/body_models/smpl_faces.npy.
Only `shapedirs` is a chumpy object in this file; it is replaced by its plain array value.
Run it with this project's Python so the pickled scipy objects match the installed scipy.
"""
import pickle
import sys
from pathlib import Path

import numpy as np


class _ChumpyStub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.state = state


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        return _ChumpyStub if module.startswith("chumpy") else super().find_class(module, name)


model = _Unpickler(open(sys.argv[1], "rb"), encoding="latin1").load()
for key, value in model.items():
    if isinstance(value, _ChumpyStub):
        model[key] = np.array(value.state["x"])

out = Path("data/body_models")
(out / "smpl").mkdir(parents=True, exist_ok=True)
pickle.dump(model, open(out / "smpl/SMPL_NEUTRAL.pkl", "wb"), protocol=4)
np.save(out / "smpl_faces.npy", model["f"].astype(np.int64))
print("wrote", out / "smpl/SMPL_NEUTRAL.pkl", "shapedirs", model["shapedirs"].shape)

#!/bin/zsh
# Builds the from-source dependencies for a CPU-only (Apple Silicon) setup. Log: build.log
set -e
cd "$(dirname "$0")"
PY="$PWD/.venv/bin/python"
export MAX_JOBS=8 FORCE_CUDA=0 MMCV_WITH_OPS=1
export CC=clang CXX=clang++ MACOSX_DEPLOYMENT_TARGET=13.0
# Apple clang 21 rejects a std specialization in the PyTorch 2.1 headers; it is only a diagnostic
export CFLAGS="-Wno-invalid-specialization" CXXFLAGS="-Wno-invalid-specialization" CPPFLAGS="-Wno-invalid-specialization"

step() { echo "\n===== [$(date +%H:%M:%S)] $1 ====="; }

step "mmcv-full 1.7.0 (with CPU ops; mmpose 0.29 requires <=1.7.0)"
[ -d third_party/mmcv ] || git clone -q --depth 1 -b v1.7.0 https://github.com/open-mmlab/mmcv.git third_party/mmcv
# mmcv 1.7.0's Apple GPU (MPS) kernels link against a symbol PyTorch 2.1 doesn't export; add a switch to skip them
grep -q MMCV_NO_MPS third_party/mmcv/setup.py || sed -i '' "309s/elif (hasattr(torch.backends, 'mps')/elif os.getenv('MMCV_NO_MPS', '0') != '1' and (hasattr(torch.backends, 'mps')/" third_party/mmcv/setup.py
export MMCV_NO_MPS=1
# PyTorch 2.1 headers need C++17; mmcv 1.7.0's CPU-only build path still asks for C++14
sed -i '' 's/-std=c++14/-std=c++17/g' third_party/mmcv/setup.py
uv pip install -p "$PY" --no-build-isolation -e third_party/mmcv
"$PY" -c "import mmcv, mmcv.ops; print('mmcv', mmcv.__version__, 'ops OK')"

step "PyTorch3D 0.7.5"
[ -d third_party/pytorch3d ] || git clone -q --depth 1 -b v0.7.5 https://github.com/facebookresearch/pytorch3d.git third_party/pytorch3d
uv pip install -p "$PY" --no-build-isolation -e third_party/pytorch3d
"$PY" -c "import pytorch3d, pytorch3d.renderer; print('pytorch3d', pytorch3d.__version__, 'OK')"

step "detectron2 + DensePose"
[ -d third_party/detectron2 ] || git clone -q --depth 1 https://github.com/facebookresearch/detectron2.git third_party/detectron2
uv pip install -p "$PY" --no-build-isolation -e third_party/detectron2
uv pip install -p "$PY" --no-build-isolation -e third_party/detectron2/projects/DensePose
"$PY" -c "import detectron2, densepose; print('detectron2', detectron2.__version__, 'densepose OK')"

step "ALL BUILDS DONE"

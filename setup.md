### Setup mmhuman3d
```bash
# Create and activate conda environment.
conda create -n vcs python=3.8 -y
conda activate vcs

# Install ffmpeg.
conda install ffmpeg

# Install PyTorch.
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Build Pytorch3D from source (important!).
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install .
cd ..

# Build mmcv-full from source.
git clone https://github.com/open-mmlab/mmcv.git -b v1.5.3
cd mmcv
MMCV_WITH_OPS=1 pip install -e . 
cd ..
pip install "mmdet<=2.25.1"
pip install "mmpose<=0.28.1"
pip install pycocotools
git clone https://github.com/open-mmlab/mmtracking.git
cd mmtracking

# Remove pycocotools from requirements/runtime.txt before proceeding (important!).
pip install .
cd ..

# Resolve dependency conflicts.
pip uninstall scipy
pip install 'scipy<=1.7.3'
pip uninstall scikit-image
pip install scikit-image==0.19.3

# Install mmhumand3d.
git clone https://github.com/open-mmlab/mmhuman3d.git
cd mmhuman3d
pip install -v -e .
```


### Setup DensePose
```bash
# Install detectron2.
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install densepose.
pip install git+https://github.com/facebookresearch/detectron2@main#subdirectory=projects/DensePose
git clone https://github.com/facebookresearch/detectron2.git
```

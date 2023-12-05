# Virtual Correspondences

Implementation of [Ma, Wei-Chiu, et al. "Virtual correspondence: Humans as a cue for extreme-view geometry." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.](https://arxiv.org/abs/2206.08365). 

### Setup
1. See [setup.md](setup.md)
2. Download [this data folder](https://drive.google.com/drive/folders/1cBX_s1n3QuUhzu8Rky-f9s_Q3GIDo3Qo?usp=sharing) and [this model checkpoints folder](https://drive.google.com/file/d/1HTHo8IpoGyvYYP6eXPa3kE-q97ZLowyL/view?usp=sharing) and put them in the root directory of this repo.

### Inference
```bash
python main.py --config config.yaml --image1_path <path to image 1> --image2_path <path to image 2>
```

### Data
Example images and virtual correspondences .npy files can be found [here](https://drive.google.com/drive/folders/111erVXmn1jEJZGistYOyjolIpbFYAvry?usp=sharing).
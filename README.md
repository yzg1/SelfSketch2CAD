# SelfSketch2CAD

Official implementation of **SelfSketch2CAD: Self-Supervised CAD Sequence Learning from Single-View Sketches**.

SelfSketch2CAD reconstructs editable CAD geometry from a single-view sketch. It combines sketch, depth, and normal features and predicts CAD primitives using self-supervised SDF reconstruction.

## Requirements

The code is written in Python and requires a CUDA-enabled NVIDIA GPU.

```bash
conda create -n selfsketch2cad python=3.10
conda activate selfsketch2cad
conda install -c conda-forge pythonocc-core
pip install torch torchvision numpy h5py pillow tqdm tensorboardX \
    opencv-python trimesh PyMCubes matplotlib shapely pyquaternion
```

## Data preparation

Following CAPRI-Net, the paper uses 5,000 ABC models for training and 1,000 for testing. Set the following paths in `exp_log/ABC/specs.json`:

```json
{
  "DataSource": "/path/to/ABC_data",
  "DinoFeaturesDir": "/path/to/precomputed_dino_features"
}
```

`DataSource` should contain:

```text
ABC_data/
|-- sdf_train.hdf5
|-- sdf_test.hdf5
|-- train_names.npz
|-- test_names.npz
`-- renderingimg/
    |-- edgemap/<model_id>/*.png
    |-- contours/<model_id>/*.png
    `-- PhotoSketch/<model_id>/*.png
```

The code expects precomputed DINO-V2 global and local features for the sketch, depth, and normal branches.

## Usage

Run all commands from the project root.

### 1. Train

```bash
python train.py --experiment ABC --gpu 0
```

Checkpoints are saved to `exp_log/ABC/ModelParameters/`. To resume training:

```bash
python train.py --experiment ABC --gpu 0 --resume 500 --target-epochs 800
```

### 2. Fine-tune a test shape

```bash
python fine-tuning.py --experiment ABC --checkpoint best \
    --input /path/to/00000003.png --epochs 50 --gpu 0
```

The first 8 characters of the image filename must be a model ID present in `test_names.npz`. A directory of PNG/JPG images can also be passed to `--input`.

### 3. Reconstruct CAD geometry

Fine-tuning must be completed before testing:

```bash
python test.py --experiment ABC --input /path/to/00000003.png \
    --grid_sample 256 --mc_threshold 0 --gpu 0
```

Results are written to:

```text
exp_log/ABC/Reconstructions/
|-- CAD/   # CAD results (.brep and .stl)
|-- MC/    # Marching Cubes meshes (.obj)
`-- sk/    # Predicted 2D sketches
```

> **Important:** in the current release, `fine-tuning.py` and `test.py` use the input filename to identify an existing test model and load its precomputed features. Direct inference on an arbitrary new sketch requires an additional preprocessing pipeline for depth/normal prediction and DINO-V2 feature extraction.

## Optional: depth and normal prediction

After placing a trained U-Net checkpoint in `DN_Checkpoints/`, run:

```bash
python depth_normal.py --mode test --checkpoint best.pth \
    --input /path/to/sketches --output ./DN_pre --gpu 0
```

Predicted maps are saved in `DN_pre/depth/` and `DN_pre/normal/`.

## Acknowledgement

This project builds on ideas and code from [DeepSDF](https://github.com/facebookresearch/DeepSDF), [SECAD-Net](https://github.com/bunnysocrazy/secad-net), [CAPRI-Net](https://github.com/FENGGENYU/CAPRI-Net), and [DINO-V2](https://github.com/facebookresearch/dinov2).

## Citation

If this project is useful in your research, please cite the accompanying SelfSketch2CAD paper.

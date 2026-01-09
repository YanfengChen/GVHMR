# Install

## Environment

```bash
git clone https://github.com/zju3dv/GVHMR
cd GVHMR

conda create -y -n gvhmr python=3.10
conda activate gvhmr
pip install -r requirements.txt
pip install -e .
# to install gvhmr in other repo as editable, try adding "python.analysis.extraPaths": ["path/to/your/package"] to settings.json
```

### Note: RTX 50xx / Blackwell (sm_120)

The pinned `torch==2.3.0+cu121` wheels in `requirements.txt` do **not** include `sm_120` kernels, so on RTX 50xx GPUs you will see errors like `no kernel image is available for execution on the device`.

To keep everything else close to the repo pins, you can:

Option A (recommended): use the provided Blackwell requirements file:

```bash
pip install --no-build-isolation -r requirements_blackwell.txt
```

Option B: upgrade in-place after installing `requirements.txt`:

```bash
# 1) Install the repo requirements first
pip install -r requirements.txt

# 2) Upgrade torch/torchvision to CUDA 12.8 wheels (includes sm_120)
pip install -U --extra-index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1+cu128 torchvision==0.24.1+cu128

# 3) Rebuild pytorch3d for your torch/CUDA stack
pip uninstall -y pytorch3d || true
conda install -y -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8
TORCH_CUDA_ARCH_LIST="12.0" CUDA_HOME="$CONDA_PREFIX" \
    pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Optional: DPVO (not recommended if you want fast inference speed)
```bash
cd third-party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip

# NOTE (torch>=2.9): DPVO upstream uses some deprecated PyTorch C++ APIs.
# This repo includes small compatibility patches:
# - dpvo/altcorr/correlation_kernel.cu: use tensor.scalar_type() in AT_DISPATCH
# - dpvo/lietorch/include/dispatch.h: update dtype dispatch for torch 2.9+
# If you update/replace the DPVO sources, you may need to re-apply these.

# torch-scatter must match your torch/CUDA build.
# For RTX 50xx / Blackwell (sm_120) + torch 2.9.1+cu128, build from source:
CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST="12.0" FORCE_CUDA=1 \
    pip install --no-build-isolation --no-binary torch-scatter torch-scatter

pip install numba pypose

# Build DPVO CUDA extensions (uses nvcc from your current toolchain).
CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST="12.0" FORCE_CUDA=1 \
    pip install -e .
```

## Inputs & Outputs

```bash
mkdir inputs
mkdir outputs
```

**Weights**

```bash
mkdir -p inputs/checkpoints

# 1. You need to sign up for downloading [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/). And the checkpoints should be placed in the following structure:

inputs/checkpoints/
├── body_models/smplx/
│   └── SMPLX_{GENDER}.npz # SMPLX (We predict SMPLX params + evaluation)
└── body_models/smpl/
    └── SMPL_{GENDER}.pkl  # SMPL (rendering and evaluation)

# 2. Download other pretrained models from Google-Drive (By downloading, you agree to the corresponding licences): https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link

inputs/checkpoints/
├── dpvo/
│   └── dpvo.pth
├── gvhmr/
│   └── gvhmr_siga24_release.ckpt
├── hmr2/
│   └── epoch=10-step=25000.ckpt
├── vitpose/
│   └── vitpose-h-multi-coco.pth
└── yolo/
    └── yolov8x.pt
```

**Data**

We provide preprocessed data for training and evaluation.
Note that we do not intend to distribute the original datasets, and you need to download them (annotation, videos, etc.) from the original websites.
*We're unable to provide the original data due to the license restrictions.*
By downloading the preprocessed data, you agree to the original dataset's terms of use and use the data for research purposes only.

You can download them from [Google-Drive](https://drive.google.com/drive/folders/10sEef1V_tULzddFxzCmDUpsIqfv7eP-P?usp=drive_link). Please place them in the "inputs" folder and execute the following commands:

```bash
cd inputs
# Train
tar -xzvf AMASS_hmr4d_support.tar.gz
tar -xzvf BEDLAM_hmr4d_support.tar.gz
tar -xzvf H36M_hmr4d_support.tar.gz
# Test
tar -xzvf 3DPW_hmr4d_support.tar.gz
tar -xzvf EMDB_hmr4d_support.tar.gz
tar -xzvf RICH_hmr4d_support.tar.gz

# The folder structure should be like this:
inputs/
├── AMASS/hmr4d_support/
├── BEDLAM/hmr4d_support/
├── H36M/hmr4d_support/
├── 3DPW/hmr4d_support/
├── EMDB/hmr4d_support/
└── RICH/hmr4d_support/
```

# <img src="assets/jovox_logo.svg" height="44" align="top" alt="JoVox logo"> JoVox: Joint Hierarchical Flow Matching for Lightweight Voxel Generation

<a href="{{PAPER_URL}}"><img src="https://img.shields.io/badge/Paper-Arxiv-b31b1b.svg" alt="Paper"></a>
<a href="https://huggingface.co/{{HF_REPO}}"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow" alt="Hugging Face"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-{{LICENSE_BADGE}}-green" alt="License"></a>

<table>
  <tr>
    <td align="center"><b>Level by level</b></td>
    <td align="center"><b>JoVox: all levels jointly</b></td>
  </tr>
  <tr>
    <td><img src="assets/level_by_level.webp" width="380" alt="Level-by-level generation"></td>
    <td><img src="assets/ours.webp" width="380" alt="JoVox generates all levels jointly"></td>
  </tr>
</table>

**JoVox** is a discrete flow matching framework that learns the **joint hierarchical distribution** of voxels while keeping generation lightweight.
This repository releases the image-conditioned **64³** generator with its training, sampling and evaluation code.
An image-to-mesh script turns the generated structure into a mesh with the second stage of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2).


## ✨ Features

### 1. Joint Hierarchical Generation
One model generates all levels, **4³ → 16³ → 64³**, from a single image.

### 2. Lightweight Training
Compared with TRELLIS.2 with the same backbone:
*   **8×** training throughput
*   **1/15** FLOPs per forward pass (0.82 vs. 12.2 TFLOPs)
*   **4 vs. 32** H100 GPUs to train

### 3. Accuracy
On 1000 Toys4k objects (image-conditioned), the Chamfer distance is **11% lower** than that of TRELLIS.2.

| Method | CD_V ↓ | IoU ↑ | CD_shell ↓ | FID_N ↓ | Uni3D ↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| TRELLIS.2 | 0.01250 | 0.4017 | 0.524e-3 | 33.3 | 0.3865 |
| **JoVox** | **0.01107** | **0.4158** | **0.407e-3** | **31.6** | **0.3868** |

### 4. Image to Mesh
Three steps: background removal, structure generation, and meshing with the second stage of TRELLIS.2.


## 🛠 Installation

### Prerequisites
- **System**: Linux.
- **Hardware**: an NVIDIA GPU (tested on **A100**).
- **Software**: [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) 12.4, [Conda](https://docs.anaconda.com/miniconda/install/#quick-command-line-install), Python 3.10, PyTorch 2.6.0.

### Installation Steps
1. Clone the repo and create the environment:
    ```sh
    git clone https://github.com/yuyujunjun/JoVox.git
    cd JoVox
    conda create -n jovox python=3.10 -y && conda activate jovox
    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt
    ```

2. Clone [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) into `third_party/TRELLIS.2` at the commit we used:
    ```sh
    git clone --recursive https://github.com/microsoft/TRELLIS.2.git third_party/TRELLIS.2
    git -C third_party/TRELLIS.2 checkout 5565d24
    ```
    Then install `flex_gemm`, `cumesh`, `nvdiffrast`, `utils3d` and `flash-attn` with its `setup.sh`.
    The code finds `third_party/TRELLIS.2` by itself.

3. Install [o-voxel-gpu](https://github.com/yuyujunjun/o-voxel-gpu) in place of the `o_voxel` that TRELLIS.2 ships:
    ```sh
    git clone --recursive https://github.com/yuyujunjun/o-voxel-gpu.git
    git -C o-voxel-gpu checkout fa9ae23
    pip install ./o-voxel-gpu --no-build-isolation
    ```

4. For evaluation, install [PyTorch3D](https://github.com/facebookresearch/pytorch3d) (0.7.9 tested).

The `prep` and `mesh` stages of the image-to-mesh script run in a separate TRELLIS.2 environment,
set up as in the TRELLIS.2 README and with o-voxel-gpu from step 3.


## 📦 Pretrained Weights

| Model | Resolution | Link |
| :--- | :--- | :--- |
| **JoVox** | 64³ | [Hugging Face](https://huggingface.co/{{HF_REPO}}) |

All models are read from the Hugging Face cache (`$HF_HOME`), so they can be downloaded once and copied to an offline machine:
```sh
huggingface-cli download {{HF_REPO}} model.safetensors --local-dir checkpoints/voxelgen64
huggingface-cli download facebook/dinov3-vitl16-pretrain-lvd1689m        # gated: accept the license on its page first
huggingface-cli download microsoft/TRELLIS.2-4B --revision af44b45f2e35a493886929c6d786e563ec68364d
huggingface-cli download briaai/RMBG-2.0                                 # gated: accept the license on its page first
```
DINOv3 encodes the condition image. TRELLIS.2-4B and RMBG-2.0 are used only by the image-to-mesh script.

LPIPS_V also needs two torch hub models (`$TORCH_HOME`):
```sh
python -c "import lpips, torch; lpips.LPIPS(net='alex'); torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')"
```


## 🚀 Usage

### 1. Image to Mesh
```sh
# 1. background removal and centering (TRELLIS.2 environment)
python scripts/img2mesh.py --stage prep --in_dir <images> --out <out>
# 2. 64³ voxel structure (this environment)
python scripts/img2mesh.py --stage voxel --out <out> --seed 0
# 3. mesh from TRELLIS.2's second stage (TRELLIS.2 environment)
python scripts/img2mesh.py --stage mesh --out <out> --seed 0
```
Outputs in `<out>`:
*   `prep/`: centered RGBA images (`--fill` sets how much of the canvas the object covers)
*   `voxels/`: generated structures
*   `mesh/`: one `.ply` per image

### 2. Evaluation
Each generated structure is compared with the ground truth of the same object by CD_V, IoU and LPIPS_V.
All files are under `data/`.

1. **Condition list.** `data/splits/eval_cond.csv`, one row per object, with columns `split,subset,uid,image_path,seed`.
   `split` is `test`, `image_path` is one condition image under `data/renders/<subset>/renders3/<uid>/`,
   and `seed` is the sampling seed. We used 1000 Toys4k objects.
2. **Ground truth.** Name the meshes `<uid>.glb` in one directory, then run:
    ```sh
    python scripts/voxelize_gt.py --dataset toys4k --src <mesh dir>
    python scripts/render_gt.py --dataset toys4k --method gt
    ```
    This writes the voxels and the LPIPS_V renderings to `data/gen/toys4k/gt/`.
3. **Generate.**
    ```sh
    python scripts/generate.py --spec voxelgen64 --dataset toys4k
    ```
    Keep the default batch of 8 (`--batch`) to reproduce our numbers. Batch composition slightly changes the samples.
4. **Metrics.**
    ```sh
    python scripts/evaluate.py --dataset toys4k --method voxelgen64 --compute_missing
    ```
    Results go to `data/gen/toys4k/voxelgen64/released/eval/toys4k/`.
    In `core.json`, `result.cd_p24_sum` is CD_V and `result.iou64` … `result.iou8` are the IoUs. `core.csv` lists every object.
    In `vis_pair.json`, `vis_pair.lpips` is LPIPS_V.

Add `--n N` to steps 3 and 4 to use only the first N rows.


## 🏋 Training

### 1. Data Preparation
Training reads 64³ occupancy and condition images of TRELLIS-500K objects:
```
$PROJECT_DATA/TRELLIS-500K-blender/<subset>/voxels/<uid>.vxz     64³ occupancy, Blender frame (Z up)
$PROJECT_DATA/TRELLIS-500K-blender/splits/train.csv              columns subset,uid
$PROJECT_DATA/TRELLIS-500K-blender/splits/test.csv               columns subset,uid
$VFM_IMAGE_ROOT/<subset>/renders3/<uid>/*.png                    RGBA condition images
$VFM_IMAGE_ROOT/<subset>/renders3/<uid>/transforms.json          camera poses (optional)
```
`PROJECT_DATA` defaults to `./data` and `VFM_IMAGE_ROOT` to `./data/TRELLIS-500K`.
With `transforms.json`, condition images come from views with elevation in [−10°, 30°]. Without it, all views are used.
We trained on the sketchfab, 3D-FUTURE, HSSD and ABO subsets and kept Toys4k for testing. {{RENDERING_DESCRIPTION}}

The occupancy files are made from meshes in two steps. The subset directory needs the `metadata.csv` of the
TRELLIS.2 data toolkit, with columns `sha256` and `local_path` (absolute, or relative to the working directory):
```sh
python scripts/voxelize_meshes.py --root <raw>/<subset> --max_workers 8
python scripts/to_blender_frame.py --src_root <raw>/<subset> --dst_root $PROJECT_DATA/TRELLIS-500K-blender/<subset>
```
The first step voxelizes each normalized mesh at 512³ and writes 512³ and 64³ occupancy in the glTF frame (Y up).
The second rotates both into the Blender frame of the condition renderings. Both accept `--rank/--world_size` for sharding.

### 2. Running Training
```sh
accelerate launch --config_file configs/accelerate-train.yaml scripts/train.py --config configs/voxelgen64.yaml
```
The configuration uses 4 GPUs × 64 objects (global batch 256) for 328,400 steps: 10,000 warmup steps, cosine decay,
and a final cosine anneal to 1e-6 over the last 8,400 steps. Any value can be overridden on the command line:
*   `model.num_levels`: 2 for 64³ (4³ → 16³ → 64³), 3 for 256³ (one more level, read from `<subset>/dual_grid_512/` and max-pooled to 256³).
*   `model.global_infer_root_end`: one window end per level except the last, e.g. `[0.8]` or `[0.6, 0.8]`.
*   `train.batch_size`, `train.accum_steps`: per-GPU batch and gradient accumulation.
*   `logger.trackers`: `[tensorboard]` by default. Add `wandb` and set `logger.wandb.entity` to log to Weights & Biases.

Checkpoints go to `exps/checkpoints/voxelgen/voxel_fm_64/<step>/` and logs to `exps/trackers`.
Training resumes from the latest checkpoint automatically.

To use your own checkpoint, pass the spec `voxelgen64_trained` with `--step`:
```sh
python scripts/generate.py --spec voxelgen64_trained --step 328400 --dataset toys4k
python scripts/evaluate.py --dataset toys4k --method voxelgen64_trained --step 328400 --compute_missing
python scripts/img2mesh.py --stage voxel --out <out> --spec voxelgen64_trained --step 328400
```


## 🧩 Related Packages

*   **[TRELLIS.2](https://github.com/microsoft/TRELLIS.2):**
    sparse transformer backbone, second-stage meshing, and data toolkit.
*   **[o-voxel-gpu](https://github.com/yuyujunjun/o-voxel-gpu):**
    GPU voxelization and voxel file I/O.


## ⚖ License

{{LICENSE_OF_THIS_REPOSITORY}}

The training code builds on [OpenLRM](https://github.com/3DTopia/OpenLRM) (Apache-2.0, see `LICENSE`).
`voxelgen/network/backbone.py` and the image encoder in `voxelgen/network/model.py` build on
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (MIT).
[DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) and [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
are under their own licenses (see their model pages).


## 📚 Citation

If you find this work useful, please cite:
```bibtex
{{BIBTEX}}
```

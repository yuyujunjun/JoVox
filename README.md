# <img src="assets/jovox_logo.svg" height="44" align="top" alt="JoVox logo"> JoVox: Joint Hierarchical Flow Matching for Lightweight Voxel Generation

<a href="{{PAPER_URL}}"><img src="https://img.shields.io/badge/Paper-Arxiv-b31b1b.svg" alt="Paper"></a>
<a href="https://huggingface.co/xiangjuntang/JoVox-64"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow" alt="Hugging Face"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-green" alt="License"></a>

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
    ```

2. Clone [TRELLIS.2](https://github.com/microsoft/TRELLIS.2) into `third_party/TRELLIS.2` and install its dependencies
   into the same environment. Its `setup.sh` must run on a machine with an NVIDIA GPU.
    ```sh
    git clone --recursive https://github.com/microsoft/TRELLIS.2.git third_party/TRELLIS.2
    git -C third_party/TRELLIS.2 checkout 5565d24
    cd third_party/TRELLIS.2 && . ./setup.sh --basic --flash-attn --nvdiffrast --cumesh --flexgemm && cd ../..
    ```
    `--basic` also tries `sudo apt install libjpeg-dev`. It is not needed here, so a failed `sudo` can be ignored.

3. Install [o-voxel-gpu](https://github.com/yuyujunjun/o-voxel-gpu) in place of the `o_voxel` that TRELLIS.2 ships:
    ```sh
    git clone --recursive https://github.com/yuyujunjun/o-voxel-gpu.git
    git -C o-voxel-gpu checkout fa9ae23
    pip install ./o-voxel-gpu --no-build-isolation
    ```

4. Install the remaining dependencies. Run this after step 2, which installs other versions of some of them:
    ```sh
    pip install -r requirements.txt
    ```

5. Install [PyTorch3D](https://github.com/facebookresearch/pytorch3d), used by training and evaluation. It is built from
   source on a machine with an NVIDIA GPU:
    ```sh
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9" --no-build-isolation
    ```


## 📦 Pretrained Weights

| Model | Resolution | Link |
| :--- | :--- | :--- |
| **JoVox-64** | 64³ | [Hugging Face](https://huggingface.co/xiangjuntang/JoVox-64) |

Download it together with the models that the code uses:
```sh
hf download xiangjuntang/JoVox-64 model.safetensors --local-dir checkpoints/voxelgen64
hf download facebook/dinov3-vitl16-pretrain-lvd1689m        # gated: accept the license on its page first
hf download microsoft/TRELLIS.2-4B --revision af44b45f2e35a493886929c6d786e563ec68364d
hf download briaai/RMBG-2.0                                 # gated: accept the license on its page first
python -c "import lpips, torch; lpips.LPIPS(net='alex'); torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')"
```
*   **DINOv3**: encodes the condition image.
*   **TRELLIS.2-4B**, **RMBG-2.0**: second-stage meshing and background removal, used only by Image to Mesh.
*   **LPIPS**, **DINOv2**: used only by the evaluation (LPIPS_V and DINO_V).

All of them are read from the local caches (`$HF_HOME` and `$TORCH_HOME`), so a machine without internet access
can use copies downloaded elsewhere.


## 🚀 Usage

### 1. Quick Try
`assets/examples/` holds eight condition images of Toys4k test objects. They already have a transparent background,
so instead of running the `prep` stage, copy them into `prep/`:
```sh
mkdir -p outputs/examples/prep && cp assets/examples/*.png outputs/examples/prep/
python scripts/img2mesh.py --stage voxel --out outputs/examples --seed 0
python scripts/img2mesh.py --stage mesh  --out outputs/examples --seed 0
```
The meshes are written to `outputs/examples/mesh/`.

### 2. Image to Mesh
Run the three stages on a folder of images:
```sh
python scripts/img2mesh.py --stage prep  --in_dir <images> --out <out>
python scripts/img2mesh.py --stage voxel --out <out> --seed 0
python scripts/img2mesh.py --stage mesh  --out <out> --seed 0
```
The results are in `<out>/prep/` (centered images), `<out>/voxels/` (structures) and `<out>/mesh/` (one `.ply` per image).
```sh
Usage: scripts/img2mesh.py --stage STAGE --out OUT [OPTIONS]
Options:
    --stage           prep: remove the background and center the object
                      voxel: generate the 64³ structure
                      mesh: turn the structure into a mesh with the second stage of TRELLIS.2
    --out             Output folder. Each stage reads the results of the previous one from it
    --in_dir          prep: folder of input images (.png, .jpg, .jpeg, .webp)
    --fill            prep: fraction of the image covered by the longest side of the object (default 0.74)
    --seed            voxel, mesh: random seed (default 0)
    --spec            voxel: voxelgen64 (the released model, default) or voxelgen64_trained (your checkpoint)
    --step            voxel: checkpoint step, only with voxelgen64_trained
    --batch           voxel: images per sampling call (default 1)
```

### 3. Evaluation
Compare the structures generated by the released model with the ground truth of the same objects by CD_V, IoU, LPIPS_V and DINO_V.

1. Write the list of test objects to `data/splits/eval_cond.csv`, one row per object:

    | Column | Content |
    | :--- | :--- |
    | `split` | always `test` |
    | `subset`, `uid` | the object |
    | `image_path` | its condition image, under `data/renders/<subset>/renders3/<uid>/` |
    | `seed` | sampling seed |

2. Put the ground-truth meshes in one folder as `<uid>.glb`, and voxelize and render them:
    ```sh
    python scripts/voxelize_gt.py --dataset toys4k --src <mesh folder>
    python scripts/render_gt.py --dataset toys4k
    ```

3. Generate one structure per object:
    ```sh
    python scripts/generate.py --spec voxelgen64 --dataset toys4k
    ```

4. Compute the metrics:
    ```sh
    python scripts/evaluate.py --dataset toys4k --method voxelgen64 --compute_gt_dino
    ```

```sh
Usage: scripts/generate.py --spec SPEC --dataset DATASET [OPTIONS]
Options:
    --spec            voxelgen64 (the released model) or voxelgen64_trained (your checkpoint)
    --step            Checkpoint step, only with voxelgen64_trained
    --dataset         Name of the evaluation set (toys4k). Results go to data/gen/<dataset>/<spec>/
    --n               Use only the first N objects of the list
    --batch           Images per sampling call (default 8)
    --rank            Index of this job when the list is split over several jobs (default 0)
    --world_size      Number of jobs the list is split over (default 1)

Usage: scripts/evaluate.py --dataset DATASET --method METHOD [OPTIONS]
Options:
    --dataset         Name of the evaluation set (toys4k)
    --method          The --spec given to generate.py, or gt
    --step            Checkpoint step, only with voxelgen64_trained
    --n               Use only the first N objects of the list
    --metrics         core (CD_V, IoU), vis_pair (LPIPS_V, DINO_V), or both comma-separated (default)
    --compute_gt_dino Compute and cache the ground-truth DINOv2 features used by DINO_V
                      (needed the first time a ground truth is evaluated)
    --overwrite       Recompute metrics that already exist (otherwise the script stops)
    --batch           Objects per batch for CD_V and IoU, changes only memory and speed (default 32)
    --device          Torch device (default cuda)

Usage: scripts/voxelize_gt.py --dataset DATASET --src SRC [--pattern PATTERN]
Options:
    --dataset         Name of the evaluation set (toys4k). Results go to data/gen/<dataset>/gt/
    --src             Folder of ground-truth meshes named <uid>.<ext>
    --pattern         File pattern of the meshes in --src (default *.glb)

Usage: scripts/render_gt.py --dataset DATASET [OPTIONS]
Options:
    --dataset         Name of the evaluation set (toys4k). Renders data/gen/<dataset>/gt/ from 6 views
    --rank            Index of this job when the work is split over several jobs (default 0)
    --world_size      Number of jobs the work is split over (default 1)
```

The metrics are written to `data/gen/toys4k/voxelgen64/released/eval/toys4k/`:
*   `core.json`: `result.cd_p24_sum` is CD_V, `result.iou64` to `result.iou8` are the IoUs at 64³ to 8³
*   `core.csv`: the same metrics for every object
*   `vis_pair.json`: `vis_pair.lpips` is LPIPS_V and `vis_pair.dino_cos` is DINO_V

Add `--n N` to steps 3 and 4 to use only the first N objects. The results then go to `released_n<N>/` instead of `released/`.


## 🏋 Training

### 1. Data Preparation
Training reads, for each object, its occupancy and its condition images. We trained on the sketchfab, 3D-FUTURE,
HSSD and ABO subsets of TRELLIS-500K and kept Toys4k for testing.

| Path | Content |
| :--- | :--- |
| `$PROJECT_DATA/TRELLIS-500K-blender/<subset>/voxels/<uid>.vxz` | 64³ occupancy |
| `$PROJECT_DATA/TRELLIS-500K-blender/splits/{train,test}.csv` | objects of each split (columns `subset,uid`) |
| `$VFM_IMAGE_ROOT/<subset>/renders3/<uid>/*.png` | condition images (RGBA) |
| `$VFM_IMAGE_ROOT/<subset>/renders3/<uid>/transforms.json` | camera poses (optional) |

`PROJECT_DATA` defaults to `./data` and `VFM_IMAGE_ROOT` to `./data/TRELLIS-500K`.
If `transforms.json` is present, conditions are drawn from views with an elevation between −10° and 30° when there are any.

The occupancy is made from the meshes in two steps:
```sh
python scripts/voxelize_meshes.py --root <raw>/<subset> --max_workers 8
python scripts/to_blender_frame.py --src_root <raw>/<subset> --dst_root $PROJECT_DATA/TRELLIS-500K-blender/<subset>
```
```sh
Usage: scripts/voxelize_meshes.py --root ROOT [OPTIONS]
Voxelizes every mesh listed in ROOT/metadata.csv (the file the TRELLIS.2 data toolkit writes) at 512³ and 64³.
Options:
    --root            Subset folder containing metadata.csv (columns sha256, local_path)
    --instances       Voxelize only these sha256 ids (a file with one per line, or comma-separated)
    --resolution      Resolutions to write, the first must be 512 (default 512,64)
    --max_workers     Meshes processed in parallel (default 8)
    --rank            Index of this job when the work is split over several jobs (default 0)
    --world_size      Number of jobs the work is split over (default 1)

Usage: scripts/to_blender_frame.py --src_root SRC --dst_root DST [OPTIONS]
Rotates the occupancy from Y-up to Z-up, the orientation of the condition images.
Options:
    --src_root        Subset folder written by voxelize_meshes.py (with dual_grid_512/ and voxels/)
    --dst_root        Output folder, different from --src_root
    --max_workers     Files processed in parallel (default 8)
    --rank            Index of this job when the work is split over several jobs (default 0)
    --world_size      Number of jobs the work is split over (default 1)
```

### 2. Running Training
```sh
accelerate launch --config_file configs/accelerate-train.yaml scripts/train.py --config configs/voxelgen64.yaml
```
The default configuration trains for 328,400 steps on 4 GPUs with 64 objects per GPU.
For another number of GPUs, set `num_processes` in `configs/accelerate-train.yaml` (and `distributed_type: "NO"` for one GPU).
```sh
Usage: scripts/train.py --config CONFIG [KEY=VALUE ...]
Options:
    --config                      Configuration file (configs/voxelgen64.yaml)
    KEY=VALUE                     Override any value of the configuration. The ones you are most likely to change:
      train.batch_size            Objects per GPU (default 64). Keep batch_size × GPUs × accum_steps at 256
      train.accum_steps           Gradient accumulation steps (default 1)
      model.num_levels            2: generate 64³ (default). 3: add one level and generate 256³,
                                  reading the 512³ occupancy in <subset>/dual_grid_512/
      model.global_infer_root_end Fraction of sampling after which each coarser level is complete.
                                  [0.8] for 2 levels (default), [0.6,0.8] for 3
      logger.trackers             [tensorboard] (default), or [tensorboard,wandb] with logger.wandb.entity=<entity>
```

Checkpoints are saved to `exps/checkpoints/voxelgen/voxel_fm_64/<step>/`, and training resumes from the latest one.

### 3. Using Your Checkpoint
The commands above use the released model (`voxelgen64`). To use a checkpoint of your own training instead,
name it `voxelgen64_trained` and give its step:
```sh
python scripts/img2mesh.py --stage voxel --out <out> --spec voxelgen64_trained --step 328400          # Image to Mesh
python scripts/generate.py --spec voxelgen64_trained --step 328400 --dataset toys4k                   # Evaluation, step 3
python scripts/evaluate.py --dataset toys4k --method voxelgen64_trained --step 328400 --compute_gt_dino # Evaluation, step 4
```


## 🧩 Related Packages

*   **[TRELLIS.2](https://github.com/microsoft/TRELLIS.2):**
    sparse transformer backbone, second-stage meshing, and data toolkit.
*   **[o-voxel-gpu](https://github.com/yuyujunjun/o-voxel-gpu):**
    GPU voxelization and voxel file I/O.


## ⚖ License

The code is released under the **[Apache-2.0 License](LICENSE)**.

The training code builds on [OpenLRM](https://github.com/3DTopia/OpenLRM) (Apache-2.0).
`voxelgen/network/backbone.py` and the image encoder in `voxelgen/network/model.py` build on
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (MIT).
[DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) and [RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0)
are under their own licenses (see their model pages).


## 📚 Citation

If you find this work useful, please cite:
```bibtex
{{BIBTEX}}
```

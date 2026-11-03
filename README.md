<p align="center"><img src="assets/jovox_logo.svg" width="220" alt="JoVox logo"></p>

# JoVox: Joint Hierarchical Flow Matching for Lightweight Voxel Generation

JoVox is a discrete flow matching framework that learns the joint hierarchical distribution of voxels while keeping generation lightweight. · [Paper]({{PAPER_URL}}) · [Weights](https://huggingface.co/{{HF_REPO}})

This repository contains the image-conditioned 64³ voxel structure generator: training, sampling,
evaluation (CD_V, IoU, LPIPS_V), and an image-to-mesh script that decodes the generated structure
with the second stage of [TRELLIS.2](https://github.com/microsoft/TRELLIS.2).

## Installation

Tested with Python 3.10, PyTorch 2.6.0 and CUDA 12.4 on NVIDIA A100 GPUs.

```bash
conda create -n voxelgen python=3.10 -y && conda activate voxelgen
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

The sparse transformer comes from [TRELLIS.2](https://github.com/microsoft/TRELLIS.2). Clone it into
`third_party/TRELLIS.2` at the commit we used, and install `flex_gemm`, `cumesh`, `nvdiffrast`, `utils3d` and
`flash-attn` with its setup script into the same environment. The code finds `third_party/TRELLIS.2` by itself:

```bash
git clone --recursive https://github.com/microsoft/TRELLIS.2.git third_party/TRELLIS.2
git -C third_party/TRELLIS.2 checkout 5565d24
```

Voxelization uses the GPU version of `o_voxel`, [o-voxel-gpu](https://github.com/yuyujunjun/o-voxel-gpu).
Install it in place of the `o_voxel` that TRELLIS.2 ships:

```bash
git clone --recursive https://github.com/yuyujunjun/o-voxel-gpu.git
git -C o-voxel-gpu checkout fa9ae23
pip install ./o-voxel-gpu --no-build-isolation
```

The evaluation additionally needs [PyTorch3D](https://github.com/facebookresearch/pytorch3d) (0.7.9 tested).
Weights & Biases logging is optional (`pip install wandb`).

The `prep` and `mesh` stages of the image-to-mesh script run TRELLIS.2 itself. We run them in a separate
environment set up as the TRELLIS.2 README describes, with o-voxel-gpu installed in place of its `o_voxel` as above.

### Pretrained models

All models are read from the Hugging Face cache (`$HF_HOME`, default `~/.cache/huggingface`), so they can be
downloaded once on a machine with internet access and copied to an offline cluster.

```bash
huggingface-cli download {{HF_REPO}} model.safetensors --local-dir checkpoints/voxelgen64
huggingface-cli download facebook/dinov3-vitl16-pretrain-lvd1689m        # gated: accept the license on its page first
huggingface-cli download microsoft/TRELLIS.2-4B --revision af44b45f2e35a493886929c6d786e563ec68364d
huggingface-cli download briaai/RMBG-2.0                                 # gated: accept the license on its page first
```

`checkpoints/voxelgen64/model.safetensors` is the released generator (EMA weights). DINOv3 encodes the
condition image. TRELLIS.2-4B and RMBG-2.0 are only used by the image-to-mesh script (the second stage and
background removal).

The evaluation (LPIPS_V) also uses weights from the torch hub cache (`$TORCH_HOME`, default `~/.cache/torch`).
Running this once fetches them:

```bash
python -c "import lpips, torch; lpips.LPIPS(net='alex'); torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')"
```

## Image to mesh

```bash
# 1. background removal and centering (TRELLIS.2 environment)
python scripts/img2mesh.py --stage prep --in_dir <images> --out <out>
# 2. 64³ voxel structure (this environment)
python scripts/img2mesh.py --stage voxel --out <out> --seed 0
# 3. mesh from TRELLIS.2's second stage (TRELLIS.2 environment)
python scripts/img2mesh.py --stage mesh --out <out> --seed 0
```

Each stage writes into `<out>`: `prep/` (centered RGBA images), `voxels/` (the generated structures),
`mesh/` (one `.ply` per image). `--fill` sets how much of the canvas the object covers in step 1.

## Evaluation

Evaluation compares each generated structure with the ground-truth structure of the same object and reports
CD_V, IoU and LPIPS_V as defined in the paper. Everything is read from and written to `data/`.

1. **Condition list.** `data/splits/eval_cond.csv` with one row per object and the columns
   `split,subset,uid,image_path,seed`. `split` is `test` for every row; `image_path` is one condition
   image of the object, under `data/renders/<subset>/renders3/<uid>/`; `seed` is the sampling seed of
   that object. Pick any set of objects you like (we used 1000 Toys4k objects).
2. **Ground truth.** Put the meshes in one directory as `<uid>.glb`, then
   ```bash
   python scripts/voxelize_gt.py --dataset toys4k --src <mesh dir>
   python scripts/render_gt.py --dataset toys4k --method gt
   ```
   This writes `data/gen/toys4k/gt/` (voxels and the renderings used by LPIPS_V).
3. **Generate.**
   ```bash
   python scripts/generate.py --spec voxelgen64 --dataset toys4k
   ```
   Samples are generated in batches of 8 (`--batch`). Batch composition changes the samples slightly, so keep
   the default to reproduce our numbers exactly.
4. **Metrics.**
   ```bash
   python scripts/evaluate.py --dataset toys4k --method voxelgen64 --compute_missing
   ```
   Results are written to `data/gen/toys4k/voxelgen64/released/eval/toys4k/`: `core.json`
   (`result.cd_p24_sum` is CD_V, `result.iou64` … `result.iou8` are the IoUs; `core.csv` has every object)
   and `vis_pair.json` (`vis_pair.lpips` is LPIPS_V).

Add `--n N` to steps 3 and 4 to use only the first N rows of the list.

## Training

### Data

Training reads 64³ occupancy and condition images of TRELLIS-500K objects:

```
$PROJECT_DATA/TRELLIS-500K-blender/<subset>/voxels/<uid>.vxz     64³ occupancy, Blender frame (Z up)
$PROJECT_DATA/TRELLIS-500K-blender/splits/train.csv              columns subset,uid
$PROJECT_DATA/TRELLIS-500K-blender/splits/test.csv               columns subset,uid
$VFM_IMAGE_ROOT/<subset>/renders3/<uid>/*.png                    RGBA condition images
$VFM_IMAGE_ROOT/<subset>/renders3/<uid>/transforms.json          camera poses (optional)
```

`PROJECT_DATA` defaults to `./data` and `VFM_IMAGE_ROOT` to `./data/TRELLIS-500K`.
When `transforms.json` is present, condition images are drawn from views with elevation between
−10° and 30°; otherwise from all views. We trained on the sketchfab, 3D-FUTURE, HSSD and ABO subsets
and kept Toys4k for testing. {{RENDERING_DESCRIPTION}}

The occupancy files are made from the meshes in two steps. The subset directory must contain the
`metadata.csv` of the TRELLIS.2 data toolkit (columns `sha256`, `local_path`; `local_path` absolute or
relative to the working directory):

```bash
python scripts/voxelize_meshes.py --root <raw>/<subset> --max_workers 8
python scripts/to_blender_frame.py --src_root <raw>/<subset> --dst_root $PROJECT_DATA/TRELLIS-500K-blender/<subset>
```

The first step voxelizes each normalized mesh at 512³ and writes 512³ and 64³ occupancy in the glTF frame
(Y up); the second rotates both to the Blender frame used by TRELLIS and by the condition renderings.
Both steps support `--rank/--world_size` for sharding.

### Run

```bash
accelerate launch --config_file configs/accelerate-train.yaml scripts/train.py --config configs/voxelgen64.yaml
```

The configuration trains on 4 GPUs with 64 objects per GPU (global batch 256) for 328,400 steps: a
10,000-step warmup, a cosine schedule, and a final cosine anneal to 1e-6 over the last 8,400 steps.
Any configuration value can be overridden on the command line, e.g. `train.batch_size=32`.
`model.num_levels` sets the depth of the hierarchy: 2 generates 64³ (4³ → 16³ → 64³), 3 generates 256³
(one more level, read from `<subset>/dual_grid_512/` and max-pooled to 256³); `model.global_infer_root_end`
then takes one window end per level except the last, e.g. `[0.6, 0.8]`.
Checkpoints are written to `exps/checkpoints/voxelgen/voxel_fm_64/<step>/`.
TensorBoard logs go to `exps/trackers`; to also log to Weights & Biases, set
`logger.trackers=[tensorboard,wandb]` and `logger.wandb.entity=<your entity>`.

To sample or evaluate your own checkpoint, use the spec `voxelgen64_trained` with `--step`:

```bash
python scripts/generate.py --spec voxelgen64_trained --step 328400 --dataset toys4k
python scripts/evaluate.py --dataset toys4k --method voxelgen64_trained --step 328400 --compute_missing
python scripts/img2mesh.py --stage voxel --out <out> --spec voxelgen64_trained --step 328400
```

## License

{{LICENSE_OF_THIS_REPOSITORY}}

This repository builds on [OpenLRM](https://github.com/3DTopia/OpenLRM) (Apache-2.0, see `LICENSE`).
`voxelgen/network/backbone.py` and the image encoder in `voxelgen/network/model.py` build on
[TRELLIS.2](https://github.com/microsoft/TRELLIS.2) (MIT).

## Citation

```bibtex
{{BIBTEX}}
```

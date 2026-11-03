import dataclasses
import os
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file

from voxelgen.network.model import VoxelNetFM
from voxelgen.train.ema import EMA_FILE

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclasses.dataclass
class VoxelGenSpec:

    name: str
    config_rel: str
    ckpt_rel: str
    step: str | None = None
    weights: str = "model"

    def ckpt_file(self) -> str:
        d = os.path.join(REPO, self.ckpt_rel)
        if self.step is not None:
            return os.path.join(d, self.step, "model.safetensors")
        return os.path.join(d, "model.safetensors")


SPECS = {
    "voxelgen64": VoxelGenSpec(
        name="voxelgen64", config_rel="configs/voxelgen64.yaml", ckpt_rel="checkpoints/voxelgen64"),
    "voxelgen64_trained": VoxelGenSpec(
        name="voxelgen64_trained", config_rel="configs/voxelgen64.yaml",
        ckpt_rel="exps/checkpoints/voxelgen/voxel_fm_64", weights="ema"),
}


def get_spec(name: str, step=None) -> VoxelGenSpec:
    assert name in SPECS, f"unknown spec {name}; choose from {sorted(SPECS)}"
    spec = SPECS[name]
    assert (step is None) == (spec.weights == "model"), \
        f"{name}: --step is {'not used' if spec.weights == 'model' else 'required'}"
    return spec if step is None else dataclasses.replace(spec, step=f"{int(step):06d}")


class VoxelGenerator:

    def __init__(self, spec: VoxelGenSpec, dev):
        self.spec = spec
        self.cfg = OmegaConf.load(os.path.join(REPO, spec.config_rel))
        self.device = str(dev)
        self.sampling_kwargs = dict(self.cfg.sampling)
        print(f"[voxelgen] {spec.name} sampling parameters: {self.sampling_kwargs}", flush=True)

        model = VoxelNetFM(**self.cfg.model)
        self.native_res = 2 ** model.native_level
        ckpt = spec.ckpt_file()
        assert os.path.isfile(ckpt), f"checkpoint not found: {ckpt}"
        missing, unexpected = model.load_state_dict(load_file(ckpt), strict=False)
        assert not missing, f"these keys are empty after loading: {missing[:5]} ({len(missing)} in total)"
        if spec.weights == "ema":
            ema_f = os.path.join(os.path.dirname(ckpt), EMA_FILE)
            assert os.path.isfile(ema_f), f"{ema_f} not found"
            ema = load_file(ema_f)
            assert all(v.dtype == torch.float32 for v in ema.values()), f"{ema_f} contains non-fp32 tensors"
            params = dict(model.named_parameters())
            extra = [k for k in ema if k not in params]
            assert not extra, f"EMA weights contain parameters not in the model: {extra[:5]}"
            lack = [n for n, p in params.items() if p.requires_grad and n not in ema]
            assert not lack, f"trainable parameters missing from the EMA weights: {lack[:5]}"
            with torch.no_grad():
                for k, v in ema.items():
                    params[k].copy_(v.to(params[k].dtype))
            print(f"[voxelgen] {spec.name}: loaded {len(ema)} trainable tensors from {ema_f} "
                  f"(EMA weights)", flush=True)
        self.model = model.to(dev).eval()
        print(f"[voxelgen] {spec.name} checkpoint {ckpt}; "
              f"missing {len(missing)} / unexpected {len(unexpected)}; "
              f"native resolution {self.native_res}^3", flush=True)

    @torch.no_grad()
    def generate(self, images: torch.Tensor, masks: torch.Tensor, seed: int) -> List[np.ndarray]:
        out = self.model.sampling(images=images.to(self.device), masks=masks.to(self.device),
                                  n_samples=len(images), seed=seed, **self.sampling_kwargs)
        assert len(out) == len(images), f"expected {len(images)} samples, sampler returned {len(out)}"
        return out

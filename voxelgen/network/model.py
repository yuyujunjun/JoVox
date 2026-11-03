from typing import List

import numpy as np
import torch
import torch.nn as nn

from trellis2.modules.image_feature_extractor import DinoV3FeatureExtractor

from .backbone import HierVoxelFlow
from .hierarchy import NUM_CLASSES, MASK_INDEX, ROOT_LEVEL, SLOT_OCTANT, _key, _unkey, native_level_of
from .sampler import _PyramidSpec, _CondBoundNet, sample
from .schedule import GlobalSchedule, check_truth_ml, iid_snapshots_ml

P_UNCOND = 0.1


class ImageEncoder(DinoV3FeatureExtractor):

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        assert isinstance(image, torch.Tensor) and image.ndim == 4, "image must be a batched tensor (B, C, H, W)"
        image = torch.nn.functional.interpolate(image, size=(self.image_size, self.image_size), mode='bilinear',
                                                align_corners=False, antialias=True)
        image = self.transform(image).cuda()
        return self.extract_features(image)


class VoxelNetFM(nn.Module):

    _NOT_SAVED = ("image_cond_model",)

    @property
    def flow(self) -> HierVoxelFlow:
        return self.sparse_structure_flow_model

    def __init__(self, fm_model_channels: int, fm_num_blocks: int, fm_num_heads: int,
                 fm_cond_channels: int, fm_use_checkpoint: bool, point_embed_freqs: int,
                 global_schedule: dict, global_infer_root_end: list, num_levels: int):
        super().__init__()
        self.image_cond_model = ImageEncoder(model_name="facebook/dinov3-vitl16-pretrain-lvd1689m")
        self.sparse_structure_flow_model = HierVoxelFlow(
            model_channels=int(fm_model_channels), cond_channels=int(fm_cond_channels),
            num_blocks=int(fm_num_blocks), num_heads=int(fm_num_heads),
            use_checkpoint=bool(fm_use_checkpoint), point_embed_freqs=int(point_embed_freqs),
            num_levels=int(num_levels))
        print("sparse_structure_flow_model loaded")
        import nvdiffrast.torch as dr
        self.glctx = dr.RasterizeCudaContext()

        self.root_level = ROOT_LEVEL
        self.num_levels = int(num_levels)
        self.native_level = native_level_of(self.num_levels)
        assert self.flow.model_channels == fm_model_channels, (
            f"backbone width {self.flow.model_channels} != config {fm_model_channels}")
        self._pyr_spec = _PyramidSpec(self.num_levels)
        self.register_buffer("_fm_step", torch.zeros((), dtype=torch.long))
        print(f"[voxelfm] condition dropout p_uncond={P_UNCOND}", flush=True)
        print(f"[voxelfm] backbone {fm_model_channels}x{fm_num_blocks}x{fm_num_heads}"
              f" | {self.num_levels} levels ({2**ROOT_LEVEL}^3 -> {2**self.native_level}^3)", flush=True)

        from flow_matching.path import MixtureDiscreteProbPath
        from flow_matching.path.scheduler import PolynomialConvexScheduler
        from flow_matching.loss import MixturePathGeneralizedKL
        self._slot_gkl = MixturePathGeneralizedKL(
            path=MixtureDiscreteProbPath(scheduler=PolynomialConvexScheduler(n=1.0)),
            reduction='none')

        gs = dict(global_schedule)
        keys = ('power', 'root_end_center', 'root_end_lo', 'root_end_hi', 'root_end_point_mass',
                'steps_lo', 'steps_hi', 'cleanup_steps', 'remask_steps')
        assert sorted(gs) == sorted(keys), f'global_schedule keys must be {keys}, got {sorted(gs)}'
        self._global_sched = GlobalSchedule(**gs)
        ends = tuple(float(x) for x in global_infer_root_end)
        assert len(ends) == self.num_levels - 1 and all(a < b for a, b in zip(ends, ends[1:])), \
            f'{self.num_levels} levels need {self.num_levels - 1} increasing window ends, got {ends}'
        assert all(self._global_sched.e_lo <= e <= self._global_sched.e_hi for e in ends), \
            'the inference window ends must be within the training range'
        self._global_infer_root_end = ends
        self.flow._global_sched = self._global_sched
        self.flow._global_infer_root_end = self._global_infer_root_end
        print(f"[voxelfm] schedule: {self._global_sched.describe()} | inference window ends "
              f"{self._global_infer_root_end} (last level 1) | loss: per-slot GKL on masked units, mean per level then over levels",
              flush=True)

    def state_dict(self, **kwargs):
        state_dict = super().state_dict(**kwargs)
        return {k: v for k, v in state_dict.items() if not k.startswith(self._NOT_SAVED)}

    def load_state_dict(self, state_dict, strict=True, assign=False):
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith(self._NOT_SAVED)}
        result = super().load_state_dict(filtered_state_dict, strict=False)
        missing_keys = [k for k in result.missing_keys if not any(p in k for p in self._NOT_SAVED)]
        unexpected_keys = [k for k in result.unexpected_keys if not any(p in k for p in self._NOT_SAVED)]
        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError(f"incompatible checkpoint: missing {missing_keys}, unexpected {unexpected_keys}")
        return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)

    @torch.no_grad()
    def _build_cond(self, images, masks) -> dict:
        images, masks = images[:, 0], masks[:, 0]
        resolution = images.shape[-1]
        assert images.shape[-1] == images.shape[-2]
        scale = min(1, 1024 / resolution)
        if scale < 1:
            images = torch.nn.functional.interpolate(
                images, size=(int(resolution * scale), int(resolution * scale)),
                mode='bilinear', align_corners=False, antialias=True)
        images = images * masks
        self.image_cond_model.to(images.device)
        self.image_cond_model.image_size = 512
        cond = self.image_cond_model(images)
        return {'cond': cond, 'neg_cond': torch.zeros_like(cond)}

    @staticmethod
    def _cond_dropout(cond, neg_cond):
        mask = list(np.random.rand(cond.shape[0]) < P_UNCOND)
        mask = torch.tensor(mask, device=cond.device).reshape(-1, *[1] * (cond.ndim - 1))
        return torch.where(mask, neg_cond, cond)

    def _stage_tokens(self, stages, li: int):
        assert len(stages) == 2 * self.num_levels, \
            f'expected a pyramid with {2 * self.num_levels} levels, got {len(stages)}'
        p = self.root_level + 2 * li
        pk, _ = stages[2 * li]
        ck, clab = stages[2 * li + 1]
        assert len(ck) > 0 and bool(np.all(np.diff(ck) > 0)), \
            'child cell keys must be non-empty and strictly sorted'
        cxyz = ((_unkey(pk, p)[:, None, :] << 1) + SLOT_OCTANT[None]).reshape(-1, 3)
        ckeys = _key(cxyz, p + 1)
        idx = np.clip(np.searchsorted(ck, ckeys), 0, len(ck) - 1)
        hit = ck[idx] == ckeys
        slots = np.where(hit, clab[idx], 0).reshape(-1, 8).astype(np.int64)
        assert int(hit.sum()) == len(ck), \
            f'level {li}: only {int(hit.sum())} of {len(ck)} occupied child cells map to slots'
        if li > 0:
            assert bool((slots != 0).any(1).all()), f'level {li}: occupied parent cell with all 8 slots empty'
        return pk, slots, p

    def forward(self, voxel_pyramid=None, images=None, masks=None, **kwargs):
        assert voxel_pyramid is not None, "voxel_pyramid is required (set dataset.load_voxel_occ)"
        sched = self._global_sched
        dev = next(self.flow.parameters()).device
        step = int(self._fm_step.item())
        self._fm_step += 1
        rng = np.random.default_rng(step * 1000003 + int(dev.index or 0))
        L = self.num_levels
        co, st, lb, lv, rr = [], [], [], [], []
        truths = []
        for stages in voxel_pyramid:
            levels = []
            for li in range(L):
                pk, lab, p = self._stage_tokens(stages, li)
                assert p == self.root_level + 2 * li, (li, p)
                levels.append((pk, lab))
            assert np.array_equal(levels[0][0], np.arange(64))
            truths.append(check_truth_ml(levels))
        snaps = iid_snapshots_ml(truths, sched.p, rng)
        for b, (lvs, r_b) in enumerate(snaps):
            rr.append(r_b)
            for lev, (keys, state, lab) in enumerate(lvs):
                if len(keys) == 0:
                    continue
                co.append(np.concatenate([np.full((len(keys), 1), b, dtype=np.int64),
                                          _unkey(keys, self.root_level + 2 * lev)], 1))
                st.append(state.astype(np.int64)); lb.append(lab.astype(np.int64))
                lv.append(np.full(len(keys), lev, dtype=np.int64))
        T = lambda a, dt=torch.long: torch.from_numpy(np.concatenate(a)).to(device=dev, dtype=dt)
        coords, state, labels, lvl = T(co).int(), T(st), T(lb), T(lv)
        cond = self._cond_dropout(**self._build_cond(images, masks))
        r = torch.tensor(rr, dtype=torch.float32, device=dev)
        logits = self.flow(coords, state, r, lvl, cond=cond)
        N_ = coords.shape[0]
        assert logits.shape == (N_, 8, NUM_CLASSES), tuple(logits.shape)
        assert bool(torch.isfinite(logits).all()), 'non-finite backbone outputs'
        known = state != MASK_INDEX
        assert bool((state[known] == labels[known]).all()), 'known slots differ from the ground truth'
        t_tok = r[coords[:, 0].long(), lvl]
        t_slot = t_tok.repeat_interleave(8)
        ok = t_slot < 1.0
        lg = torch.cat([logits.float(), torch.full_like(logits[..., :1].float(), float('-inf'))], -1)
        lslot = self._slot_gkl(logits=lg.reshape(N_ * 8, 1, -1), x_1=labels.reshape(N_ * 8, 1),
                               x_t=state.reshape(N_ * 8, 1), t=t_slot.clamp(max=1.0 - 1e-6)).view(-1)
        assert bool(torch.isfinite(lslot[ok]).all()), 'non-finite per-slot GKL loss'
        loss, per_level = self._reduce_levels(lslot, lvl.repeat_interleave(8), sel=ok)
        return {"loss": loss, "mse": loss.detach(), **per_level}

    def _reduce_levels(self, ltok, lvl, sel):
        out, means = {}, []
        for i in range(self.num_levels):
            m = (lvl == i) & sel
            if not bool(m.any()):
                out[f"ce_level{i}"] = ltok.new_tensor(float('nan'))
                continue
            v = ltok[m].mean()
            means.append(v)
            out[f"ce_level{i}"] = v.detach()
        assert means, 'no tokens in any level'
        return torch.stack(means).mean(), out

    @torch.no_grad()
    def sampling(self, images, masks, n_samples: int, seed: int,
                 steps: int, cfg_strength: float, temperature: float) -> List[np.ndarray]:
        cond = self._build_cond(images, masks)
        for k in ('cond', 'neg_cond'):
            c = cond[k]
            if c.shape[0] != n_samples:
                assert c.shape[0] == 1, (
                    f"condition {k} has batch size {c.shape[0]}, expected {n_samples} or 1")
                cond[k] = c.expand(n_samples, *c.shape[1:]).contiguous()
        dev = next(self.flow.parameters()).device
        assert cfg_strength >= 1.0, f"cfg_strength must be >= 1, got {cfg_strength}"
        keys = sample(_CondBoundNet(self.flow, cond, cfg_strength), self._pyr_spec,
                      n_samples=n_samples, k=int(steps), schedule_power=self._global_sched.p,
                      seed=int(seed), device=str(dev), verbose=False,
                      sched=self._global_sched, root_end=self._global_infer_root_end,
                      temperature=float(temperature))
        return [_unkey(np.asarray(k, dtype=np.int64), self.native_level) for k in keys]

    @staticmethod
    def _coords_hash(coords_xyz: torch.Tensor, base: int = 2048) -> torch.Tensor:
        coords_xyz = coords_xyz.long()
        return coords_xyz[:, 0] + coords_xyz[:, 1] * base + coords_xyz[:, 2] * base * base

    def _coords_metrics(self, pred_coords: torch.Tensor, gt_coords: torch.Tensor, batch_size: int) -> dict:
        eps = 1e-8
        precision_list = []
        recall_list = []
        f1_list = []

        for b in range(batch_size):
            pred_b = pred_coords[pred_coords[:, 0] == b][:, 1:]
            gt_b = gt_coords[gt_coords[:, 0] == b][:, 1:]

            if pred_b.numel() == 0 and gt_b.numel() == 0:
                precision_list.append(torch.tensor(1.0, device=pred_coords.device))
                recall_list.append(torch.tensor(1.0, device=pred_coords.device))
                f1_list.append(torch.tensor(1.0, device=pred_coords.device))
                continue
            if pred_b.numel() == 0 or gt_b.numel() == 0:
                precision_list.append(torch.tensor(0.0, device=pred_coords.device))
                recall_list.append(torch.tensor(0.0, device=pred_coords.device))
                f1_list.append(torch.tensor(0.0, device=pred_coords.device))
                continue

            pred_hash = torch.unique(self._coords_hash(pred_b))
            gt_hash = torch.unique(self._coords_hash(gt_b))
            inter = torch.isin(pred_hash, gt_hash).sum().float()

            precision = inter / (pred_hash.numel() + eps)
            recall = inter / (gt_hash.numel() + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

        return {
            'precision': torch.stack(precision_list).mean(),
            'recall': torch.stack(recall_list).mean(),
            'f1': torch.stack(f1_list).mean(),
        }

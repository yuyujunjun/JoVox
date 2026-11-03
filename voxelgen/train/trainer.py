# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from OpenLRM (https://github.com/3DTopia/OpenLRM).

import argparse
import gc
import logging
import math
import os
import random
import re
import shutil
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator, DataLoaderConfiguration
from accelerate.logging import get_logger
from accelerate.tracking import WandBTracker
from accelerate.utils import (DistributedDataParallelKwargs, InitProcessGroupKwargs,
                              ProjectConfiguration, set_seed)
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from voxelgen.network.model import VoxelNetFM
from voxelgen.interfaces.render import make_projection_mtx, render_voxel_pbr
from .dataset import BalancedResumableSampler, build_dataloader
from .ema import EMA_FILE, ParamEMA
from .lr_scheduler import CosineWarmupScheduler
from .optim import AdamW

logger = get_logger(__name__)


def parse_configs():
    parser = argparse.ArgumentParser(
        description="Train the generator. Any configuration value can be overridden after the options as key=value, "
                    "e.g. train.batch_size=32.")
    parser.add_argument('--config', type=str, required=True, help="configuration file, e.g. configs/voxelgen64.yaml")
    args, unknown = parser.parse_known_args()
    return OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_cli(unknown))


class _TqdmStreamHandler(logging.StreamHandler):
    def emit(self, record):
        tqdm.write(self.format(record))


def configure_logger(stream_level, log_level, file_path=None):
    formatter = logging.Formatter("[%(asctime)s] %(name)s: [%(levelname)s] %(message)s")
    stream_handler = _TqdmStreamHandler()
    stream_handler.setLevel(stream_level.upper())
    stream_handler.setFormatter(formatter)
    project_logger = logging.getLogger(__name__.split('.')[0])
    project_logger.setLevel(log_level.upper())
    project_logger.addHandler(stream_handler)
    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(log_level.upper())
        file_handler.setFormatter(formatter)
        project_logger.addHandler(file_handler)


def _wandb_key(tag: str) -> str:
    m = re.fullmatch(r"(?P<metric>[^/]+)/(?P<split>train|val)/(?:step|epoch)", tag)
    if m:
        metric, split = m["metric"], m["split"]
        if split == "val":
            return f"val/{metric}"
        return f"loss/{metric}" if metric in ("loss", "mse") else f"misc/{metric}"
    m = re.fullmatch(r"Images_split/(?P<split>[^/]+)/(?P<x>.+)", tag)
    if m:
        return f"images/{m['split']}/{m['x']}"
    return tag


class _WandBSectioned(WandBTracker):

    _axis_defined = False

    def _define_axis(self):
        if not self._axis_defined and getattr(self, "run", None) is not None:
            self.run.define_metric("global_step")
            self._axis_defined = True

    def _log(self, values: dict, step, **kwargs):
        self._define_axis()
        d = {_wandb_key(k): v for k, v in values.items()}
        s = 0 if step is None else int(step)
        d["global_step"] = s
        super().log(d, step=s, **kwargs)

    def log(self, values: dict, step=None, **kwargs):
        self._log(values, step, **kwargs)

    def log_images(self, values: dict, step=None, **kwargs):
        import wandb

        def _to_wandb(img):
            t = img.detach().float().clamp(0, 1)
            if t.ndim == 3 and t.shape[0] in (1, 3, 4):
                t = t.permute(1, 2, 0)
            return wandb.Image((t * 255).to(torch.uint8).cpu().numpy())

        self._log({k: [_to_wandb(img) for img in v] for k, v in values.items()}, step, **kwargs)


class _DummyProfiler:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def step(self):
        pass


def to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, list):
        return [to_device(v, device) for v in data]
    if isinstance(data, tuple):
        return tuple(to_device(v, device) for v in data)
    return data


def _coords_metrics(pred_coords: torch.Tensor, gt_coords: torch.Tensor) -> dict:
    eps = 1e-8
    pred_b, gt_b = pred_coords[:, 1:], gt_coords[:, 1:]
    dev = pred_coords.device
    if pred_b.numel() == 0 and gt_b.numel() == 0:
        one = torch.tensor(1.0, device=dev)
        return {'precision': one, 'recall': one, 'f1': one}
    if pred_b.numel() == 0 or gt_b.numel() == 0:
        zero = torch.tensor(0.0, device=dev)
        return {'precision': zero, 'recall': zero, 'f1': zero}

    def h(c):
        c = c.long()
        return torch.unique(c[:, 0] + c[:, 1] * 2048 + c[:, 2] * 2048 * 2048)
    pred_hash, gt_hash = h(pred_b), h(gt_b)
    inter = torch.isin(pred_hash, gt_hash).sum().float()
    precision = inter / (pred_hash.numel() + eps)
    recall = inter / (gt_hash.numel() + eps)
    return {'precision': precision, 'recall': recall,
            'f1': 2 * precision * recall / (precision + recall + eps)}


def _pose24_align(pred_c, gt_c, native):
    from voxelgen.evaluate.paired import pose24_batch, pad_clouds, centers
    dev = pred_c.device
    gt = gt_c[:, 1:].cpu().numpy().astype(np.int64)
    pr = pred_c[:, 1:].cpu().numpy().astype(np.int64)
    Q, LQ = pad_clouds([centers(gt)], dev)
    names, best_cd, cd_raw, best_int = pose24_batch([pr], Q, LQ, dev)
    rot = torch.as_tensor(best_int[0], dtype=pred_c.dtype, device=dev)
    out = torch.cat([pred_c[:, :1], rot], 1)
    assert out.shape == pred_c.shape
    return out, names[0], best_cd[0], cd_raw[0]


class Trainer:

    def __init__(self):
        self.cfg = parse_configs()
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.accelerator = Accelerator(
            mixed_precision=self.cfg.train.mixed_precision,
            gradient_accumulation_steps=self.cfg.train.accum_steps,
            log_with=self._trackers(),
            project_config=ProjectConfiguration(logging_dir=self.cfg.logger.tracker_root),
            dataloader_config=DataLoaderConfiguration(use_seedable_sampler=True, split_batches=False),
            kwargs_handlers=[
                DistributedDataParallelKwargs(find_unused_parameters=self.cfg.train.find_unused_parameters),
                InitProcessGroupKwargs(timeout=timedelta(seconds=int(self.cfg.train.dist_timeout_s))),
            ],
        )
        if self.accelerator.num_processes > 1:
            print(f"[dist] collective timeout = {int(self.cfg.train.dist_timeout_s)} s", flush=True)
        set_seed(self.cfg.experiment.seed, device_specific=True)
        with self.accelerator.main_process_first():
            configure_logger(
                stream_level=self.cfg.logger.stream_level,
                log_level=self.cfg.logger.log_level,
                file_path=os.path.join(
                    self.cfg.logger.log_root,
                    self.cfg.experiment.parent, self.cfg.experiment.child,
                    f"{self.timestamp}.log",
                ) if self.accelerator.is_main_process else None,
            )
        logger.info(self.accelerator.state, main_process_only=False, in_order=True)

        self.N_max_global_steps: int = None
        self.N_global_steps_per_epoch: int = None
        self.global_step: int = 0
        self.current_epoch: int = 0
        self.model = VoxelNetFM(**self.cfg.model)
        self.optimizer = self._build_optimizer(self.model, self.cfg)
        self.train_loader = build_dataloader(self.cfg, 'train', balanced=False)
        self.val_loader = build_dataloader(self.cfg, 'val', balanced=False)
        self.scheduler = self._build_scheduler(self.optimizer, self.cfg)

    def _trackers(self):
        out = []
        for name in self.cfg.logger.trackers:
            if name != "wandb":
                out.append(name)
                continue
            w = self.cfg.logger.get("wandb", {}) or {}
            out.append(_WandBSectioned(
                run_name=w.get("project", "voxelgen"),
                entity=w.get("entity", None),
                name=self.cfg.experiment.child,
                group=self.cfg.experiment.parent,
                dir=self.cfg.logger.tracker_root,
                resume="allow",
                id=w.get("id", None) or self.cfg.experiment.child,
            ))
        return tuple(out)

    def __enter__(self):
        self.accelerator.init_trackers(
            project_name=os.path.join(self.cfg.experiment.parent, self.cfg.experiment.child))
        self.prepare_everything()
        self.log_initial_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.accelerator.end_training()

    @staticmethod
    def control(option: str = None, synchronized: bool = False):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                if option is None or hasattr(self.accelerator, option):
                    accelerated_func = getattr(self.accelerator, option)(func) if option is not None else func
                    result = accelerated_func(self, *args, **kwargs)
                    if synchronized:
                        self.accelerator.wait_for_everyone()
                    return result
                raise AttributeError(f"Accelerator has no attribute {option}")
            return wrapper
        return decorator

    @property
    def device(self):
        return self.accelerator.device

    @property
    def global_step_in_epoch(self):
        return self.global_step % self.N_global_steps_per_epoch

    def _build_optimizer(self, model, cfg):
        no_decay_params = []
        for name, module in model.named_modules():
            if isinstance(module, nn.LayerNorm):
                no_decay_params.extend([p for p in module.parameters()])
            elif hasattr(module, 'bias') and module.bias is not None:
                no_decay_params.append(module.bias)
        _no_decay_ids = set(map(id, no_decay_params))
        decay_params = [p for p in model.parameters() if id(p) not in _no_decay_ids]
        decay_params = list(filter(lambda p: p.requires_grad, decay_params))
        no_decay_params = list(filter(lambda p: p.requires_grad, no_decay_params))
        logger.info(f"======== Weight decay parameters: {len(decay_params)}, without weight decay: {len(no_decay_params)} ========")
        opt_groups = [
            {'params': decay_params, 'weight_decay': cfg.train.optim.weight_decay, "lr": cfg.train.optim.lr},
            {'params': no_decay_params, 'weight_decay': 0.0, "lr": cfg.train.optim.lr},
        ]
        return AdamW(opt_groups, lr=cfg.train.optim.lr, betas=(cfg.train.optim.beta1, cfg.train.optim.beta2))

    def _build_scheduler(self, optimizer, cfg):
        local_batches_per_epoch = math.floor(len(self.train_loader) / self.accelerator.num_processes)
        total_global_batches = cfg.train.epochs * math.ceil(local_batches_per_epoch / self.cfg.train.accum_steps)
        warmup = cfg.train.scheduler.warmup_real_iters
        sched_max = int(cfg.train.scheduler.decay_iters)
        min_lr = float(cfg.train.scheduler.min_lr)
        assert sched_max > warmup, f"decay horizon {sched_max} must be larger than warmup {warmup}"
        fa = cfg.train.scheduler.final_anneal
        print(f"[sched] cosine decay horizon={sched_max} | total steps={total_global_batches}"
              f" | warmup={warmup} | min_lr={min_lr:g} | "
              + (f"final anneal from {fa.start} over {fa.steps} steps to {fa.min_lr:g}" if fa else "no final anneal"),
              flush=True)
        return CosineWarmupScheduler(
            optimizer=optimizer, warmup_iters=warmup, max_iters=sched_max, min_lr=min_lr,
            final_anneal_start=int(fa.start) if fa else None, final_anneal_iters=int(fa.steps) if fa else 0,
            final_min_lr=float(fa.min_lr) if fa else 0.0)

    def prepare_everything(self):
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader)
        self.accelerator.register_for_checkpointing(self.scheduler)
        N_total_batch_size = self.cfg.train.batch_size * self.accelerator.num_processes * self.cfg.train.accum_steps
        self.N_global_steps_per_epoch = math.ceil(len(self.train_loader) / self.cfg.train.accum_steps)
        self.N_max_global_steps = int(self.cfg.train.max_steps)
        logger.info(f"======== N_max_global_steps: {self.N_max_global_steps} | N_total_batch_size: {N_total_batch_size}"
                    f" | N_epochs: {self.cfg.train.epochs} | N_global_steps_per_epoch: {self.N_global_steps_per_epoch} ========")
        logger.info(f"======== Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)} ========")
        self.accelerator.wait_for_everyone()
        self.ema = ParamEMA(self.accelerator.unwrap_model(self.model), float(self.cfg.train.ema_decay))
        self.load_ckpt_or_auto_resume_(self.cfg)
        torch.cuda.empty_cache()

    def auto_resume_(self, cfg) -> bool:
        ckpt_root = os.path.join(cfg.saver.checkpoint_root, cfg.experiment.parent, cfg.experiment.child)
        if not os.path.exists(ckpt_root):
            return False
        ckpt_dirs = os.listdir(ckpt_root)
        if len(ckpt_dirs) == 0:
            return False

        def extract_iter(f):
            match = re.search(r'(\d+)', f)
            return int(match.group(1)) if match else -1
        latest_ckpt = str(sorted(ckpt_dirs, key=extract_iter)[-1])
        latest_ckpt_dir = os.path.join(ckpt_root, latest_ckpt)
        logger.info(f"======== Auto-resume from {latest_ckpt_dir} ========")
        self.accelerator.load_state(latest_ckpt_dir, strict=True)
        ema_f = os.path.join(latest_ckpt_dir, EMA_FILE)
        if os.path.isfile(ema_f):
            from safetensors.torch import load_file
            self.ema.load_state_dict(load_file(ema_f), self.accelerator.unwrap_model(self.model))
            logger.info(f"======== Loaded EMA weights from {ema_f} ========")
        else:
            self.ema.reset_from(self.accelerator.unwrap_model(self.model))
            logger.info(f"======== No {EMA_FILE} in the checkpoint; EMA weights reset to the resumed weights ========")
        gc.collect()
        torch.cuda.empty_cache()
        self.global_step = int(latest_ckpt)
        self.current_epoch = self.global_step // self.N_global_steps_per_epoch
        return True

    @control(synchronized=True)
    def load_ckpt_or_auto_resume_(self, cfg):
        if cfg.saver.auto_resume and self.auto_resume_(cfg):
            return
        logger.debug("======== No checkpoint or model is loaded ========")

    @staticmethod
    def _save_manual(obj, f, save_on_each_node=False, safe_serialization=False):
        from collections import OrderedDict
        from functools import partial
        from accelerate.state import PartialState
        from accelerate.utils.other import clean_state_dict_for_safetensors
        import safetensors.torch
        if safe_serialization:
            save_func = partial(safetensors.torch.save_file, metadata={"format": "pt"})
            if isinstance(obj, OrderedDict):
                obj = clean_state_dict_for_safetensors(obj)
        else:
            save_func = torch.save
        if PartialState().is_main_process and not save_on_each_node:
            save_func(obj, f)
        elif PartialState().is_local_main_process and save_on_each_node:
            save_func(obj, f)

    def _save_state_manual(self, output_dir):
        from pathlib import Path
        from accelerate.checkpointing import save_custom_state
        from accelerate.utils import SAFE_WEIGHTS_NAME, OPTIMIZER_NAME, SCHEDULER_NAME, SCALER_NAME, RNG_STATE_NAME
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving current state to {output_dir}")
        on_each_node = self.accelerator.project_configuration.save_on_each_node
        out = Path(output_dir)

        weights = [self.accelerator.get_state_dict(m, unwrap=False) for m in self.accelerator._models]
        for hook in self.accelerator._save_model_state_pre_hook.values():
            hook(self.accelerator._models, weights, output_dir)
        for i, state in enumerate(weights):
            name = SAFE_WEIGHTS_NAME if i == 0 else SAFE_WEIGHTS_NAME.replace(".", f"_{i}.")
            self._save_manual(state, out.joinpath(name), save_on_each_node=on_each_node, safe_serialization=True)
            logger.info(f"Model weights saved in {out.joinpath(name)}")
        del weights
        gc.collect()
        torch.cuda.empty_cache()

        for i, opt in enumerate(self.accelerator._optimizers):
            name = f"{OPTIMIZER_NAME}.bin" if i == 0 else f"{OPTIMIZER_NAME}_{i}.bin"
            self._save_manual(opt.state_dict(), out.joinpath(name), save_on_each_node=on_each_node)
            logger.info(f"Optimizer state saved in {out.joinpath(name)}")
        gc.collect()
        torch.cuda.empty_cache()

        for i, scheduler in enumerate(self.accelerator._schedulers):
            name = f"{SCHEDULER_NAME}.bin" if i == 0 else f"{SCHEDULER_NAME}_{i}.bin"
            self._save_manual(scheduler.state_dict(), out.joinpath(name), save_on_each_node=on_each_node)
            logger.info(f"Scheduler state saved in {out.joinpath(name)}")

        if self.accelerator.scaler is not None:
            torch.save(self.accelerator.scaler.state_dict(), out.joinpath(SCALER_NAME))

        states = {"step": self.global_step,
                  "random_state": random.getstate(),
                  "numpy_random_seed": np.random.get_state(),
                  "torch_manual_seed": torch.get_rng_state(),
                  "torch_cuda_manual_seed": torch.cuda.get_rng_state_all()}
        states_file = out.joinpath(f"{RNG_STATE_NAME}_{self.accelerator.state.process_index}.pkl")
        torch.save(states, states_file)
        logger.info(f"Random states saved in {states_file}")

        for i, obj in enumerate(self.accelerator._custom_objects):
            save_custom_state(obj, output_dir, i, save_on_each_node=on_each_node)
        gc.collect()
        torch.cuda.empty_cache()
        self.accelerator.project_configuration.iteration += 1

    @control('on_main_process', synchronized=True)
    def save_checkpoint(self, remove=True):
        ckpt_dir = os.path.join(self.cfg.saver.checkpoint_root, self.cfg.experiment.parent,
                                self.cfg.experiment.child, f"{self.global_step:06d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        self._save_state_manual(ckpt_dir)
        if self.accelerator.is_main_process:
            from safetensors.torch import save_file
            save_file({k: v.detach().cpu() for k, v in self.ema.state_dict().items()},
                      os.path.join(ckpt_dir, EMA_FILE))
        logger.info(f"======== Saved checkpoint at global step {self.global_step} ========")
        ckpt_dirs = sorted(os.listdir(os.path.dirname(ckpt_dir)), key=int)
        max_ckpt = int(ckpt_dirs[-1])
        ckpt_base = int(self.cfg.saver.checkpoint_keep_level)
        ckpt_period = self.cfg.saver.checkpoint_global_steps
        if max_ckpt < ckpt_period:
            return
        cur_order = ckpt_base ** math.floor(math.log(max_ckpt // ckpt_period, ckpt_base))
        cur_idx = 0
        if remove:
            while cur_order > 0:
                cur_digit = max_ckpt // ckpt_period // cur_order % ckpt_base
                while cur_idx < len(ckpt_dirs) and int(ckpt_dirs[cur_idx]) // ckpt_period // cur_order % ckpt_base < cur_digit:
                    if int(ckpt_dirs[cur_idx]) // ckpt_period % cur_order != 0:
                        shutil.rmtree(os.path.join(os.path.dirname(ckpt_dir), ckpt_dirs[cur_idx]))
                        logger.info(f"Removed checkpoint {ckpt_dirs[cur_idx]}")
                    cur_idx += 1
                cur_order //= ckpt_base

    @control('on_main_process')
    def log_scalar_kwargs(self, step: int, split: str, **scalar_kwargs):
        for key, value in scalar_kwargs.items():
            self.accelerator.log({f'{key}/{split}/step': value}, step)

    @control('on_main_process')
    def log_images(self, values: dict, step: int):
        for tracker in self.accelerator.trackers:
            if hasattr(tracker, 'log_images'):
                tracker.log_images(values, step=step)

    @control('on_main_process')
    def log_optimizer_lr(self, step: int):
        for group_id in (0, 1):
            self.accelerator.log({f'opt/lr/{group_id}': self.optimizer.param_groups[group_id]['lr']}, step)

    @control('on_main_process')
    def log_initial_info(self):
        self.accelerator.log({'Config': "```\n" + OmegaConf.to_yaml(self.cfg) + "\n```"})
        self.accelerator.log({'Model': "```\n" + str(self.model) + "\n```"})
        self.accelerator.log({'Optimizer': "```\n" + str(self.optimizer) + "\n```"})
        self.accelerator.log({'Scheduler': "```\n" + str(self.scheduler) + "\n```"})

    def set_dataloader_sampler_epoch(self, loader, epoch):
        if isinstance(loader.sampler, BalancedResumableSampler):
            loader.sampler.set_epoch(epoch)

    def train_epoch(self, pbar: tqdm, loader: torch.utils.data.DataLoader, profiler):
        self.model.train()
        local_step_losses = []
        for data in loader:
            data = to_device(data, self.accelerator.device)
            with self.accelerator.accumulate(self.model):
                for d in data:
                    loss_dict = self.model.forward(voxel_pyramid=d['voxel_pyramid'], images=d['images'], masks=d['masks'])
                    keys = sorted(loss_dict.keys())
                    self.accelerator.backward(loss_dict['loss'])
                if self.accelerator.sync_gradients and self.cfg.train.optim.clip_grad_norm > 0.:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.cfg.train.optim.clip_grad_norm)
                self.optimizer.step()
                self.ema.update(self.accelerator.unwrap_model(self.model))
                self.optimizer.zero_grad(set_to_none=True)
                local_step_losses.append(torch.stack([loss_dict[k].detach() for k in keys]))

            if self.accelerator.sync_gradients:
                profiler.step()
                self.scheduler.step()
                self.global_step += 1
                global_step_loss = torch.nanmean(self.accelerator.gather(torch.stack(local_step_losses)), dim=0)
                loss_kwargs = {k: v.item() for k, v in zip(keys, global_step_loss.unbind())}
                loss_kwargs = {k: v for k, v in loss_kwargs.items() if not math.isnan(v)}
                if self.accelerator.is_main_process:
                    self.log_scalar_kwargs(step=self.global_step, split='train', **loss_kwargs)
                    self.log_optimizer_lr(step=self.global_step)
                local_step_losses = []
                pbar.update(1)
                description = {**loss_kwargs, 'lr': self.optimizer.param_groups[0]['lr']}
                pbar.set_description('[TRAIN STEP]' + ', '.join(
                    f'{k}={tqdm.format_num(v)}' for k, v in description.items() if not math.isnan(v)))

                if self.global_step % self.cfg.saver.checkpoint_global_steps == 0:
                    torch.cuda.empty_cache()
                    self.save_checkpoint(remove=self.cfg.saver.remove_checkpoint)

                if self.global_step % self.cfg.val.global_step_period == 0:
                    torch.cuda.empty_cache()
                    rand_epoch = random.randint(0, self.cfg.train.epochs * 100 - 1)
                    self.set_dataloader_sampler_epoch(self.val_loader, rand_epoch)
                    self.evaluate()
                    self.model.train()

                if self.global_step >= self.N_max_global_steps:
                    self.accelerator.set_trigger()
                    break

    def train(self):
        self.train_loader = build_dataloader(self.cfg, 'train', balanced=True)
        self.val_loader = build_dataloader(self.cfg, 'val', balanced=True)
        starting_local_step_in_epoch = self.global_step_in_epoch * self.cfg.train.accum_steps
        skipped_loader = self.accelerator.skip_first_batches(self.train_loader, starting_local_step_in_epoch)
        logger.info(f"======== Skipped {starting_local_step_in_epoch} local batches ========")
        self.set_dataloader_sampler_epoch(self.val_loader, 10)
        self.evaluate()
        with tqdm(range(0, self.N_max_global_steps), initial=self.global_step,
                  disable=(not self.accelerator.is_main_process)) as pbar:
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=10, warmup=10, active=100),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(
                    self.cfg.logger.tracker_root, self.cfg.experiment.parent, self.cfg.experiment.child)),
                record_shapes=True, profile_memory=True, with_stack=True,
            ) if self.cfg.logger.enable_profiler else _DummyProfiler()
            with profiler:
                self.optimizer.zero_grad()
                for epoch in range(self.current_epoch, self.cfg.train.epochs):
                    self.set_dataloader_sampler_epoch(self.train_loader, epoch)
                    self.set_dataloader_sampler_epoch(self.val_loader, epoch)
                    loader = skipped_loader or self.train_loader
                    skipped_loader = None
                    self.train_epoch(pbar=pbar, loader=loader, profiler=profiler)
                    self.current_epoch = self.global_step // self.N_global_steps_per_epoch
                    if self.accelerator.check_trigger():
                        break
            logger.info(f"======== Training finished at global step {self.global_step} ========")
            self.save_checkpoint(remove=self.cfg.saver.remove_checkpoint)
            self.evaluate()

    def run(self):
        self.train()

    @torch.no_grad()
    @torch.compiler.disable
    @control('on_main_process')
    def evaluate(self):
        self.model.eval()
        for data in tqdm(self.val_loader, disable=(not self.accelerator.is_main_process)):
            metrics, images = self.evaluate_sample(to_device(data[0], self.accelerator.device))
            break
        metrics = {k: float(v) for k, v in metrics.items()}
        self.log_scalar_kwargs(step=self.global_step, split='val', **metrics)
        logger.info(f'[VAL STEP] {self.global_step}/{self.N_max_global_steps}: '
                    + ', '.join(f'{k}={tqdm.format_num(v)}' for k, v in metrics.items() if not math.isnan(v)))
        self.log_images({f'Images_split/val/{k}': v.unsqueeze(0) for k, v in images.items()}, self.global_step)

    @torch.no_grad()
    def evaluate_sample(self, data: dict):
        import pytorch3d.loss
        model = self.accelerator.unwrap_model(self.model)
        images, masks = data['images'], data['masks']
        native = 2 ** model.native_level

        def _b0(c):
            c = torch.as_tensor(np.asarray(c), dtype=torch.int32)
            return torch.cat([torch.zeros(len(c), 1, dtype=torch.int32), c], 1)

        gt_c = _b0(data['voxel_occ_gt'][0][0]).to(images.device)
        pred_c = _b0(model.sampling(images=images[:1], masks=masks[:1], n_samples=1, seed=0,
                                    **self.cfg.sampling)[0]).to(images.device)
        mtx, orig = make_projection_mtx(images.device)
        to_img = lambda c: torch.from_numpy(
            render_voxel_pbr(c.int(), model.glctx, mtx, native, cam_pos=orig)).float().permute(2, 0, 1) / 255.0
        metrics = {}
        if len(pred_c):
            m_raw = _coords_metrics(pred_c, gt_c)
            pred_c, pose, cd24, cdraw = _pose24_align(pred_c, gt_c, native)
            metrics.update({'f1_raw': m_raw['f1'], 'pose24_identity': float(pose == '+X+Y+Z'),
                            'cd_p24': cd24, 'cd_raw': cdraw})
            m = _coords_metrics(pred_c, gt_c)
            cd = pytorch3d.loss.chamfer_distance(
                (gt_c[:, 1:].float() / native)[None], (pred_c[:, 1:].float() / native)[None],
                batch_reduction=None, point_reduction=None)[0]
            metrics.update({'f1': m['f1'], 'precision': m['precision'], 'recall': m['recall'],
                            'chamfer': (cd[0].mean() + cd[1].mean()) * 0.5})
            pred_img = to_img(pred_c)
        else:
            metrics.update({'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'chamfer': float('nan')})
            pred_img = torch.zeros(3, 512, 512)
        return metrics, {'input': (images[0, 0] * masks[0, 0]).cpu(), 'gt': to_img(gt_c), 'pred': pred_img}

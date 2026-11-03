from typing import Dict

import torch

EMA_FILE = 'ema_fp32.safetensors'


class ParamEMA:

    def __init__(self, model: torch.nn.Module, decay: float):
        assert 0.0 < decay < 1.0, f"ema_decay must be in (0,1), got {decay}"
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            n: p.detach().float().clone()
            for n, p in model.named_parameters() if p.requires_grad}
        assert self.shadow, "no parameter requires grad"
        assert all(s.dtype == torch.float32 for s in self.shadow.values()), '[ema] EMA weights must be fp32'
        n_el = sum(t.numel() for t in self.shadow.values())
        print(f"[ema] decay={self.decay} | {len(self.shadow)} tensors "
              f"/ {n_el/1e6:.1f}M parameters | averaging window about {1/(1-self.decay):.0f} steps",
              flush=True)

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            s = self.shadow.get(n)
            assert s is not None, f"[ema] {n} is not in the EMA weights"
            s.mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def reset_from(self, model: torch.nn.Module) -> None:
        for n, p in model.named_parameters():
            if n in self.shadow:
                self.shadow[n].copy_(p.detach().float())

    def state_dict(self) -> Dict[str, torch.Tensor]:
        assert all(s.dtype == torch.float32 for s in self.shadow.values()), '[ema] EMA weights to save are not fp32'
        return self.shadow

    def load_state_dict(self, sd: Dict[str, torch.Tensor], model: torch.nn.Module) -> None:
        bad = [n for n, t in sd.items() if t.dtype != torch.float32]
        assert not bad, f'[ema] loaded EMA weights are not fp32: {bad[:3]}'
        missing = [n for n in self.shadow if n not in sd]
        cur = dict(model.named_parameters())
        for n in self.shadow:
            src = sd[n] if n in sd else cur[n].detach()
            self.shadow[n].copy_(src.to(self.shadow[n].device).float())
        if missing:
            print(f"[ema] checkpoint lacks {len(missing)} EMA tensors, initialized from the current weights", flush=True)


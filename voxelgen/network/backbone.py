import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from trellis2.modules import sparse as sp
from trellis2.models.structured_latent_flow import SLatFlowModel
from trellis2.models.sparse_structure_flow import TimestepEmbedder
from trellis2.modules.utils import manual_cast

from .hierarchy import NUM_CLASSES, MASK_INDEX, ROOT_LEVEL, native_level_of

UNIFIED_ROPE_RES = 256


class _InjectedRope(nn.Module):

    def __init__(self, head_dim: int, dim: int, rope_freq=(1.0, 10000.0)):
        super().__init__()
        self.head_dim = int(head_dim)
        self.dim = int(dim)
        self.rope_freq = tuple(rope_freq)

    def cache_key(self) -> str:
        return (f'rope_phase_{self.dim}d_freq{self.rope_freq[0]}-'
                f'{self.rope_freq[1]}_hd{self.head_dim}')

    @staticmethod
    def _rotary_embedding(x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases.unsqueeze(-2)
        return torch.view_as_real(x_rotated).reshape(
            *x_rotated.shape[:-1], -1).to(x.dtype)

    def forward(self, q, k=None):
        key = self.cache_key()
        phases = q.get_spatial_cache(key)
        assert phases is not None, (
            f'rope phases were not injected: looked for {key}, cache has '
            f'{sorted(q.get_spatial_cache().keys())}')
        assert phases.shape[0] == q.feats.shape[0] \
            and phases.shape[-1] == self.head_dim // 2, (
            f'injected phase shape {tuple(phases.shape)} does not match token count '
            f'{q.feats.shape[0]} / head_dim {self.head_dim}')
        q_embed = q.replace(self._rotary_embedding(q.feats, phases))
        if k is None:
            return q_embed
        return q_embed, k.replace(self._rotary_embedding(k.feats, phases))


def _rope_pad(ph: torch.Tensor, head_dim: int) -> torch.Tensor:
    tgt = head_dim // 2
    if ph.shape[-1] < tgt:
        pad = torch.ones(ph.shape[0], tgt - ph.shape[-1],
                         dtype=ph.dtype, device=ph.device)
        ph = torch.cat([ph, pad], dim=-1)
    return ph


class PointEmbed(nn.Module):

    def __init__(self, out_dim: int, n_freqs: int):
        super().__init__()
        assert n_freqs >= 1, n_freqs
        self.n_freqs = int(n_freqs)
        self.register_buffer(
            'freqs', 2.0 * math.pi * (2.0 ** torch.arange(self.n_freqs, dtype=torch.float32)),
            persistent=False)
        in_dim = 3 + 6 * self.n_freqs
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU(),
                                 nn.Linear(out_dim, out_dim))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        assert u.dim() == 2 and u.shape[1] == 3, f'PointEmbed expects [N,3], got {tuple(u.shape)}'
        u = u.float()
        x = u.unsqueeze(-1) * self.freqs
        feat = torch.cat([u, torch.sin(x).flatten(1), torch.cos(x).flatten(1)], dim=1)
        return self.mlp(feat)


class _TwoRatioEmbedder(nn.Module):

    def __init__(self, base: nn.Module, d: int, n: int = 2):
        super().__init__()
        assert n >= 2, n
        self.n = int(n)
        self.r0 = base
        for li in range(1, self.n):
            setattr(self, f'r{li}', TimestepEmbedder(d))
        for li in range(2, self.n):
            last = getattr(self, f'r{li}').mlp[-1]
            nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        assert t.dim() == 2 and t.shape[1] == self.n, f'progress condition must be [B,{self.n}], got {tuple(t.shape)}'
        return sum(getattr(self, f'r{li}')(t[:, li]) for li in range(self.n))


class HierVoxelFlow(SLatFlowModel):

    def __init__(self, *, model_channels: int, cond_channels: int, num_blocks: int, num_heads: int,
                 use_checkpoint: bool, point_embed_freqs: int, num_levels: int):
        self.mask_index = MASK_INDEX
        self.slot_classes = NUM_CLASSES
        super().__init__(resolution=2 ** native_level_of(num_levels), in_channels=1, out_channels=NUM_CLASSES,
                         model_channels=model_channels, cond_channels=cond_channels,
                         num_blocks=num_blocks, num_heads=num_heads, mlp_ratio=4, pe_mode="rope",
                         qk_rms_norm=True, qk_rms_norm_cross=True, dtype="bfloat16",
                         use_checkpoint=use_checkpoint)
        self.num_levels = int(num_levels)
        self.root_level = ROOT_LEVEL
        self.spans = (2,) * self.num_levels
        self.parent_levels = [ROOT_LEVEL + sum(self.spans[:i]) for i in range(self.num_levels)]

        del self.input_layer
        assert not hasattr(self, 'input_layer'), 'input_layer must be removed'

        self.base_embed = nn.Parameter(torch.zeros(self.model_channels))
        self.register_buffer('_bit_shifts', torch.arange(8), persistent=False)
        self.level_embed = nn.Embedding(self.num_levels, self.model_channels)
        self.point_embed = PointEmbed(self.model_channels, int(point_embed_freqs))
        print(f'[voxelfm] PointEmbed ({self.point_embed.n_freqs} bands, finest period '
              f'1/{2 ** (self.point_embed.n_freqs - 1)}) + rope on unified coordinates u*{UNIFIED_ROPE_RES}; '
              f'level embedding {tuple(self.level_embed.weight.shape)}', flush=True)

        nn.init.zeros_(self.base_embed)
        nn.init.normal_(self.level_embed.weight, std=0.02)

        for _b in self.blocks:
            _r = getattr(_b.self_attn, 'rope', None)
            assert _r is not None, 'block has no rope (pe_mode must be rope)'
            _b.self_attn.rope = _InjectedRope(_r.head_dim, _r.dim, _r.rope_freq)
        self._build_rope_unified()

        d = self.model_channels
        self.shared_tri_embed = nn.Parameter(torch.zeros(8, 3, d))
        self.grand_proj = nn.Linear(8 * d, d, bias=False)
        nn.init.normal_(self.shared_tri_embed, std=0.02)
        nn.init.xavier_uniform_(self.grand_proj.weight)

        self.out_layer = sp.SparseLinear(d, 8 * self.slot_classes)
        self.out_channels = 8 * self.slot_classes
        nn.init.constant_(self.out_layer.weight, 0)
        nn.init.constant_(self.out_layer.bias, 0)
        self.t_embedder = _TwoRatioEmbedder(self.t_embedder, self.model_channels, n=self.num_levels)

    def _shared_tri(self, child: torch.Tensor, grand: torch.Tensor, has_grand: torch.Tensor) -> torch.Tensor:
        N = child.shape[0]
        T = self.shared_tri_embed.reshape(24, -1)
        sh = self._bit_shifts
        assert child.shape == (N, 8) and int(child.max()) <= 2 and int(child.min()) >= 0
        e = F.embedding_bag(sh.view(1, 8) * 3 + child, T, mode='sum')
        if len(grand):
            n2 = grand.shape[0]
            assert grand.shape == (n2, 8, 8) and int(has_grand.sum()) == n2
            gg = F.embedding_bag((sh.view(1, 1, 8) * 3 + grand).reshape(n2 * 8, 8), T, mode='sum')
            e = e.index_put((torch.nonzero(has_grand).squeeze(1),),
                            e[has_grand] + self.grand_proj(gg.view(n2, -1)))
        elif self.training:
            e = e + self.grand_proj.weight.sum() * 0.0
        return (e + self.base_embed).to(self.base_embed.dtype)

    def embed_state(self, state: torch.Tensor) -> torch.Tensor:
        assert state.dim() == 2 and state.shape[1] == 8, \
            f'two-level state must be [N,8] slots, got {tuple(state.shape)}'
        assert state.numel() == 0 or (int(state.min()) >= 0
                                      and int(state.max()) <= self.mask_index), \
            'slot values must be in 0..255 or MASK_INDEX'
        unk = state == self.mask_index
        patt = torch.where(unk, torch.zeros_like(state), state)
        bits = (patt.unsqueeze(-1) >> self._bit_shifts) & 1
        tri = torch.where(unk.unsqueeze(-1), torch.full_like(bits, 2), bits)
        child = torch.where(unk, torch.full_like(state, 2), (patt > 0).long())
        return self._shared_tri(child, tri,
                                torch.ones(state.shape[0], dtype=torch.bool, device=state.device))

    def _build_rope_unified(self) -> None:
        from trellis2.modules.sparse.attention.rope import (
            SparseRotaryPositionEmbedder)
        head_dim = self.model_channels // self.num_heads
        emb3 = SparseRotaryPositionEmbedder(head_dim, 3)
        self.register_buffer('_rope_freqs', emb3.freqs.clone().float(), persistent=False)
        self._rope_fd = int(emb3.freq_dim)
        self._rope_hd = head_dim
        self._rope_key3 = self.blocks[0].self_attn.rope.cache_key()
        probe = torch.tensor([[0.5, 3.25, 100.75], [255.5, 0.0, 17.0]])
        ref = emb3._get_phases(probe.reshape(-1)).reshape(2, 3, -1)
        got = torch.polar(torch.ones(2, 3, self._rope_fd),
                          probe.unsqueeze(-1) * self._rope_freqs)
        assert torch.equal(ref, got), 'unified-coordinate phases do not match the rope _get_phases'

    def _unified_coords(self, coords: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        side = torch.tensor([float(1 << p) for p in self.parent_levels],
                            device=coords.device)[lv]
        return (coords[:, 1:].float() + 0.5) / side.unsqueeze(-1)

    def _unified_phases(self, u: torch.Tensor) -> torch.Tensor:
        x = u.float() * UNIFIED_ROPE_RES
        ph = torch.polar(torch.ones(*x.shape, self._rope_fd, device=x.device),
                         x.unsqueeze(-1) * self._rope_freqs)
        return _rope_pad(ph.reshape(len(u), -1), self._rope_hd)

    def forward(self, coords: torch.Tensor, state: torch.Tensor, r: torch.Tensor,
                level_idx: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        N = coords.shape[0]
        lv = level_idx.long()
        assert lv.shape == (N,), \
            f'per-token levels {tuple(lv.shape)} do not match token count {N}'
        feats = self.embed_state(state)
        u = self._unified_coords(coords, lv)
        feats = feats + self.point_embed(u).to(feats.dtype)
        feats = feats + self.level_embed(lv)
        x = sp.SparseTensor(feats, coords.int())
        x.register_spatial_cache(self._rope_key3, self._unified_phases(u))
        h = manual_cast(x, self.dtype)
        t_emb = manual_cast(self.t_embedder(r * 1000.0), self.dtype)
        cond = manual_cast(cond, self.dtype)
        for block in self.blocks:
            h = block(h, t_emb, cond)
        h = manual_cast(h, x.dtype)
        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h)
        return h.feats.view(h.feats.shape[0], 8, self.slot_classes)

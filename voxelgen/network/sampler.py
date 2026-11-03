from typing import List

import numpy as np
import torch

from .hierarchy import MASK_INDEX, NUM_CLASSES, ROOT_LEVEL, _key, _unkey, expand_two_level, native_level_of
from .schedule import ROOT_TOKEN_LEVEL, run


@torch.no_grad()
def sample(model, pyramid, n_samples: int, k: int, schedule_power: float, seed: int = 0,
           device: str = 'cuda', verbose: bool = True, sched=None, root_end: tuple = None,
           temperature: float = 1.0) -> List[np.ndarray]:
    greedy = float(temperature) == 0.0
    assert greedy or float(temperature) > 0, f'temperature must be >= 0, got {temperature}'
    print(f'[global] temperature T={float(temperature):g}{" (greedy)" if greedy else ""}', flush=True)
    assert sched is not None and root_end is not None, 'a GlobalSchedule and the window ends are required'
    L = len(pyramid.stages)
    assert L in (2, 3) and all(g - p == 2 for p, g in pyramid.stages) and pyramid.stages[0][0] == ROOT_TOKEN_LEVEL, \
        f'a 4^3 root with two or three +2 stages is expected, got {pyramid.stages}'
    assert float(schedule_power) == sched.p, \
        f'schedule power {schedule_power} differs from the training schedule power {sched.p}'
    ends = tuple(float(x) for x in root_end)
    assert len(ends) == L - 1, f'{L} levels need {L - 1} window ends, got {ends}'
    e = ends[0] if L == 2 else ends
    N, B = int(k), int(n_samples)
    if L == 2:
        assert sched.n_lo <= N <= sched.n_hi, f'inference steps {N} outside the training range [{sched.n_lo},{sched.n_hi}]'
    assert all(sched.e_lo <= x <= sched.e_hi for x in ends), \
        f'inference window ends {ends} outside the training range [{sched.e_lo},{sched.e_hi}]'
    gen = torch.Generator(device=device); gen.manual_seed(int(seed))
    lev_xyz = [torch.as_tensor(_unkey(np.arange(8 ** (ROOT_TOKEN_LEVEL + 2 * li)), ROOT_TOKEN_LEVEL + 2 * li),
                               device=device) for li in range(L)]
    n_fwd = [0]

    def draw(st, ex, active):
        exl = [None, ex] if L == 2 else ex
        idx = [torch.nonzero(exl[li], as_tuple=True) for li in range(1, L)]
        coords = torch.cat(
            [torch.cat([torch.arange(B, device=device).repeat_interleave(64)[:, None], lev_xyz[0].repeat(B, 1)], 1)]
            + [torch.cat([bi[:, None], lev_xyz[li][st.cells_l[li][bi, di]]], 1)
               for li, (bi, di) in enumerate(idx, 1)], 0).int()
        state = torch.cat([st.val[0].reshape(-1, 8)] + [st.val[li][bi, di] for li, (bi, di) in enumerate(idx, 1)], 0)
        lev = torch.cat([torch.zeros(B * 64, dtype=torch.long, device=device)]
                        + [torch.full((len(bi),), li, dtype=torch.long, device=device)
                           for li, (bi, di) in enumerate(idx, 1)])
        R = st.ratios(ex)
        order = torch.argsort(coords[:, 0], stable=True)
        lg = torch.empty((len(lev), 8, NUM_CLASSES), device=device)
        lg[order] = model(coords[order], state[order], R, lev[order]).float()
        assert lg.shape == (len(lev), 8, NUM_CLASSES), tuple(lg.shape)
        if greedy:
            d = lg.reshape(-1, NUM_CLASSES).argmax(-1).view(-1, 8)
        else:
            d = torch.multinomial(torch.softmax(lg.reshape(-1, NUM_CLASSES) / float(temperature), -1), 1,
                                  generator=gen).view(-1, 8)
        vals = [d[:B * 64].view(B, 64, 8)]
        o = B * 64
        for li, (bi, di) in enumerate(idx, 1):
            v = torch.full(st.val[li].shape, 0, dtype=d.dtype, device=device)
            v[bi, di] = d[o:o + len(bi)]; o += len(bi)
            vals.append(v)
        n_fwd[0] += 1
        return tuple(vals)

    st = run(B, sched, gen, [(e, N, 1.0)] * B, draw, device=device)
    if verbose:
        print(f'[global] {N} grid steps + final steps, {n_fwd[0]} forward passes; window ends {ends}', flush=True)
    last, p_last = L - 1, ROOT_TOKEN_LEVEL + 2 * (L - 1)
    ex = st.exist_all()[last].cpu().numpy(); cells = st.cells_l[last].cpu().numpy(); dval = st.val[last].cpu().numpy()
    assert (dval[ex] != MASK_INDEX).all()
    out = []
    for b in range(B):
        keys = cells[b][ex[b]]
        order = np.argsort(keys)
        if len(keys) == 0:
            out.append(np.zeros(0, np.int64)); continue
        xyz = expand_two_level(keys[order], dval[b][ex[b]][order], p_last)
        out.append(np.sort(_key(xyz, p_last + 2)) if len(xyz) else np.zeros(0, np.int64))
    return out


class _PyramidSpec:

    def __init__(self, num_levels: int):
        self.native_level = native_level_of(num_levels)
        self.stages = [(ROOT_LEVEL + 2 * i, ROOT_LEVEL + 2 * i + 2) for i in range(num_levels)]


class _CondBoundNet:

    def __init__(self, net, cond, cfg_strength: float):
        assert isinstance(cond, dict) and 'cond' in cond and 'neg_cond' in cond, \
            f"cond must be a {{'cond','neg_cond'}} dict, got {type(cond)}"
        self.net, self.cond = net, cond
        self.cfg_strength = float(cfg_strength)
        print(f"[cfg] guidance strength w={self.cfg_strength}", flush=True)

    def __call__(self, coords, state, r, level_idx):
        if self.cfg_strength == 1.0:
            return self.net(coords, state, r, level_idx,
                            cond=self.cond['cond'])
        B = int(self.cond['cond'].shape[0])
        co2 = coords.clone()
        co2[:, 0] += B
        if torch.is_tensor(level_idx) and level_idx.dim() == 1 and level_idx.shape[0] == coords.shape[0]:
            level_idx = torch.cat([level_idx, level_idx], 0)
        both = self.net(torch.cat([coords, co2], 0),
                        torch.cat([state, state], 0),
                        torch.cat([r, r], 0), level_idx,
                        cond=torch.cat([self.cond['cond'],
                                        self.cond['neg_cond']], 0))
        n = coords.shape[0]
        pos, neg = both[:n], both[n:]
        fin_p, fin_n = torch.isfinite(pos), torch.isfinite(neg)
        assert bool((fin_p == fin_n).all()), \
            "conditional and unconditional branches have different invalid masks"
        out = torch.where(fin_p, neg + self.cfg_strength * (pos - neg),
                          torch.full_like(pos, float('-inf')))
        assert not torch.isnan(out).any(), \
            "NaN after guidance"
        return out

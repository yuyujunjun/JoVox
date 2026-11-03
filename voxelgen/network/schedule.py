import numpy as np
import torch

from .hierarchy import MASK_INDEX, _key, _unkey

ROOT_TOKEN_LEVEL = 2


def octant(xyz: np.ndarray) -> np.ndarray:
    return ((xyz[:, 0] & 1) << 2) | ((xyz[:, 1] & 1) << 1) | (xyz[:, 2] & 1)


class GlobalSchedule:

    def __init__(self, power: float, root_end_center: float, root_end_lo: float, root_end_hi: float,
                 root_end_point_mass: float, steps_lo: int, steps_hi: int, cleanup_steps: int,
                 remask_steps: int):
        assert power > 0 and 0 < root_end_lo <= root_end_center <= root_end_hi <= 1.0
        assert 0.0 <= root_end_point_mass <= 1.0 and 2 <= steps_lo <= steps_hi
        assert cleanup_steps >= 1 and remask_steps >= 0
        self.p = float(power)
        self.e_c, self.e_lo, self.e_hi = float(root_end_center), float(root_end_lo), float(root_end_hi)
        self.e_pm = float(root_end_point_mass)
        self.n_lo, self.n_hi = int(steps_lo), int(steps_hi)
        self.C = int(cleanup_steps)
        self.remask = int(remask_steps)

    def describe(self) -> str:
        return (f'kappa0=min(t/e,1)^{self.p:g}, kappa1=t^{self.p:g}; training e: {self.e_pm:.0%} at {self.e_c:g}, otherwise triangular on '
                f'[{self.e_lo:g},{self.e_hi:g}] with mode {self.e_c:g}; steps N in [{self.n_lo},{self.n_hi}] uniform, '
                f'grid phase ~ U(0,1); remask {self.remask} steps; final {self.C} steps')


    def plan(self, e, N: int, phase: float) -> np.ndarray:
        ew = (float(e),) if np.ndim(e) == 0 else tuple(float(x) for x in e)
        g = np.unique(np.clip(np.concatenate([[0.0], (np.arange(1, N + 1) - 1 + phase) / N, [1.0]]), 0, 1))
        n = len(g) - 1
        t, b = g[:-1], g[1:]
        i = np.arange(n)
        s = np.where((i >= 1) & (self.remask > 0), g[np.maximum(i - self.remask, 0)], t)
        k1 = lambda x: np.clip(x, 0.0, 1.0) ** self.p
        qs, hs = [], []
        with np.errstate(divide='ignore', invalid='ignore'):
            for el in ew:
                k0 = lambda x, el=el: np.minimum(np.maximum(x, 0.0) / el, 1.0) ** self.p
                k0t, k0s = k0(t), k0(s)
                qs.append(np.where((k0t > 0) & (k0t < 1) & (s < t), k0s / k0t, 1.0))
                hs.append(np.where(k0s >= 1, 1.0, (k0(b) - k0s) / (1 - k0s)))
            k1t, k1s = k1(t), k1(s)
            qs.append(np.where((k1t > 0) & (s < t), k1s / k1t, 1.0))
            hs.append(np.where(i == n - 1, 1.0 / (self.C + 1), (k1(b) - k1s) / (1 - k1s)))
        grid = np.stack(qs + hs, 1)
        L = len(ew) + 1
        cleanup = np.concatenate([np.ones((self.C, L)), np.zeros((self.C, L - 1)),
                                  (1.0 / (self.C - np.arange(self.C)))[:, None]], 1)
        return np.concatenate([grid, cleanup], 0)


def _erase_earliest_count(known, born, m, gen):
    out = torch.zeros_like(known)
    kmax = int(m.max()) if len(m) else 0
    if kmax == 0:
        return out
    tb = torch.rand(known.shape, generator=gen, device=known.device)
    key = torch.where(known, born.float() + tb * 0.5,
                      torch.full(known.shape, float('inf'), device=known.device))
    idx = (-key).topk(kmax, dim=1).indices
    sel = (torch.arange(kmax, device=known.device)[None] < m[:, None]) & known.gather(1, idx)
    return out.scatter_(1, idx, sel)


def _pick(mask, n, gen):
    key = torch.where(mask, torch.rand(mask.shape, generator=gen, device=mask.device),
                      torch.full(mask.shape, 2.0, device=mask.device))
    kmax = int(n.max()) if len(n) else 0
    out = torch.zeros_like(mask)
    if kmax == 0:
        return out
    idx = key.topk(kmax, dim=1, largest=False).indices
    sel = (torch.arange(kmax, device=mask.device)[None] < n[:, None]) & mask.gather(1, idx)
    return out.scatter_(1, idx, sel)


def _b3_erase(known, born, q, k, gen, locked):
    m = torch.round(known.sum(1).double() * (1.0 - q.double())).long()
    elig = known & (born < k - 1) & ~locked
    return _erase_earliest_count(elig, born, m, gen)


def _b3_reveal(masked, h, active, gen):
    nm = masked.sum(1)
    n = torch.where(active & (nm > 0), torch.ceil(nm.double() * h.double()).long().clamp(min=1),
                    torch.zeros_like(nm))
    return _pick(masked, n, gen)


class State:

    def __init__(self, cells, valid):
        if torch.is_tensor(cells):
            cells, valid = [cells], [valid]
        B = cells[0].shape[0]
        dev = cells[0].device
        self.L = 1 + len(cells)
        self.cells_l, self.valid_l = [None] + list(cells), [None] + list(valid)
        shapes = [(B, 64, 8)] + [(B, c.shape[1], 8) for c in cells]
        self.val = [torch.full(s, MASK_INDEX, dtype=torch.long, device=dev) for s in shapes]
        self.born = [torch.full(s, -1, dtype=torch.long, device=dev) for s in shapes]
        self.lock = [torch.zeros(s, dtype=torch.bool, device=dev) for s in shapes]
        self._pflat, self._pbit, self._prow = [None], [None], [None]
        for li in range(1, self.L):
            c = self.cells_l[li]
            pk, slot, bit = (torch.as_tensor(x, device=dev) for x in parent_of(c.cpu().numpy().reshape(-1), li))
            pk, slot, bit = pk.view_as(c), slot.view_as(c), bit.view_as(c)
            if li == 1:
                row = pk
            else:
                pc = self.cells_l[li - 1]
                assert bool((pc[:, 1:] >= pc[:, :-1]).all()), f'cells of level {li - 1} must be sorted row-wise'
                row = torch.searchsorted(pc.contiguous(), pk.contiguous()).clamp(max=pc.shape[1] - 1)
                hit = torch.gather(pc, 1, row) == pk
                self.valid_l[li] = self.valid_l[li] & hit
            self._prow.append(row); self._pbit.append(bit); self._pflat.append(row * 8 + slot)

    def exist_level(self, li, parent_ex=None):
        pv = self.val[li - 1].view(len(self.val[li - 1]), -1).gather(1, self._pflat[li])
        known = pv != MASK_INDEX
        ex = self.valid_l[li] & known & (((torch.where(known, pv, torch.zeros_like(pv)) >> self._pbit[li]) & 1) == 1)
        if li >= 2:
            ex = ex & torch.gather(parent_ex, 1, self._prow[li])
        return ex

    def exist_all(self):
        out = [None]
        for li in range(1, self.L):
            out.append(self.exist_level(li, out[-1]))
        return out


    def _ex_list(self, ex):
        return [None, ex] if torch.is_tensor(ex) else ex

    def ratios(self, ex):
        ex = self._ex_list(ex)
        rs = [(self.val[0] != MASK_INDEX).float().mean((1, 2))]
        for li in range(1, self.L):
            dk = (self.val[li] != MASK_INDEX) & ex[li][..., None]
            rs.append(dk.sum((1, 2)).float() / (ex[li].sum(1) * 8).clamp(min=1).float())
        return torch.stack(rs, 1)

    def pending(self, ex):
        ex = self._ex_list(ex)
        B = len(self.val[0])
        p = (self.val[0] == MASK_INDEX).view(B, -1).any(1)
        for li in range(1, self.L):
            p = p | ((self.val[li] == MASK_INDEX) & ex[li][..., None]).view(B, -1).any(1)
        return p


def run(B, sched, gen, hps, draw, device='cpu'):
    plans = [sched.plan(*hp) for hp in hps]
    NL = plans[0].shape[1] // 2
    assert all(p.shape[1] == 2 * NL for p in plans), 'all samples in a batch must have the same number of levels'
    Lp = torch.tensor([len(p) for p in plans], device=device)
    K = int(Lp.max())
    pad = [1.0] * NL + [0.0] * NL
    P = torch.tensor(np.stack([np.concatenate([p, np.tile([pad], (K - len(p), 1))]) for p in plans]),
                     dtype=torch.float32, device=device)
    cells = [torch.arange(8 ** (ROOT_TOKEN_LEVEL + 2 * li), device=device).expand(B, -1).contiguous()
             for li in range(1, NL)]
    valid = [torch.ones_like(c, dtype=torch.bool) for c in cells]
    st = State(cells, valid)
    assert st.L == NL, f'cells have {st.L} levels but the schedule has {NL}'
    two = NL == 2
    for k in range(K):
        ex = st.exist_all()
        active = (k < Lp) & st.pending(ex)
        if not bool(active.any()):
            break
        cols = [torch.where(active, P[:, k, j], torch.full_like(P[:, k, j], d)) for j, d in enumerate(pad)]
        q, h = cols[:NL], cols[NL:]
        ex = [None]
        for li in range(NL):
            if li:
                ex.append(st.exist_level(li, ex[-1]))
                gone = ~ex[li][..., None]
                st.val[li] = st.val[li].masked_fill(gone, MASK_INDEX); st.born[li] = st.born[li].masked_fill(gone, -1)
                st.lock[li] = st.lock[li] & ~gone
                kn = (st.val[li] != MASK_INDEX) & ex[li][..., None]
            else:
                kn = st.val[0] != MASK_INDEX
            w = _b3_erase(kn.view(B, -1), st.born[li].view(B, -1), q[li], k, gen,
                          st.lock[li].view(B, -1)).view_as(kn)
            st.val[li] = st.val[li].masked_fill(w, MASK_INDEX); st.born[li] = st.born[li].masked_fill(w, -1)
        vals = draw(st, ex[1] if two else ex, active)
        assert len(vals) == NL, f'draw must return {NL} items, got {len(vals)}'
        ts = []
        for li in range(NL):
            masked = (st.val[li] == MASK_INDEX) if li == 0 else ((st.val[li] == MASK_INDEX) & ex[li][..., None])
            ts.append(_b3_reveal(masked.view(B, -1), h[li], active, gen).view_as(st.val[li]))
        for li, t in enumerate(ts):
            st.val[li] = torch.where(t, vals[li], st.val[li]); st.born[li] = st.born[li].masked_fill(t, k)
        single = sum(t.sum((1, 2)) for t in ts) == 1
        for li, t in enumerate(ts):
            st.lock[li] |= t & single[:, None, None]
    assert not bool(st.pending(st.exist_all()).any()), 'masked units remain after the last step'
    return st


def parent_of(keys: np.ndarray, li: int):
    T = ROOT_TOKEN_LEVEL + 2 * li
    xyz = _unkey(np.asarray(keys, np.int64), T)
    return _key(xyz >> 2, T - 2), octant(xyz >> 1), octant(xyz)


def check_truth_ml(levels: list):
    assert np.array_equal(levels[0][0], np.arange(64)) and levels[0][1].shape == (64, 8)
    for li in range(1, len(levels)):
        (pk_all, plab), (keys, lab) = levels[li - 1], levels[li]
        assert lab.shape == (len(keys), 8) and bool(np.all(np.diff(keys) > 0)), f'level {li}: keys must be sorted and labels must be [T,8]'
        pk, slot, bit = parent_of(keys, li)
        row = np.searchsorted(pk_all, pk)
        assert bool(np.all(row < len(pk_all))) and np.array_equal(pk_all[np.minimum(row, len(pk_all) - 1)], pk), \
            f'level {li}: parent token not found in level {li - 1}'
        assert bool(np.all((plab[row, slot] >> bit) & 1)), f'level {li}: parent slot bit is not set'
        n_bits = int(np.unpackbits(plab.astype(np.uint8).reshape(-1, 1), axis=1).sum())
        assert n_bits == len(keys), f'level {li - 1} has {n_bits} set bits but level {li} has {len(keys)} occupied cells'
    return levels


def iid_snapshots_ml(truths: list, power: float, rng):
    out = []
    for levels in truths:
        L = len(levels)
        rt = rng.uniform(0.0, 1.0, L) ** float(power)
        keys0, lab0 = levels[0]
        known = np.zeros(512, bool)
        known[rng.permutation(512)[:int(round(512 * rt[0]))]] = True
        known = known.reshape(64, 8)
        lv = [(keys0, np.where(known, lab0, MASK_INDEX), lab0)]
        r = [float(known.mean())]
        for li in range(1, L):
            keys, lab = levels[li]
            pkeys = lv[li - 1][0]
            pk, slot, _ = parent_of(keys, li)
            row = np.minimum(np.searchsorted(pkeys, pk), max(len(pkeys) - 1, 0))
            ex = (np.zeros(len(keys), bool) if len(pkeys) == 0
                  else (pkeys[row] == pk) & known[row, slot])
            n = 8 * int(ex.sum())
            dk = np.zeros(n, bool)
            dk[rng.permutation(n)[:int(round(n * rt[li]))]] = True
            known = dk.reshape(-1, 8)
            lv.append((keys[ex], np.where(known, lab[ex], MASK_INDEX), lab[ex]))
            r.append(float(dk.mean()) if n else 0.0)
        out.append((lv, r))
    return out



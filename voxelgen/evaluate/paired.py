import numpy as np
import torch
from pytorch3d.ops import knn_points

GRID = 64
LEVELS = (64, 32, 16, 8)


def pad_clouds(clouds, device):
    B = len(clouds); N = max(len(c) for c in clouds)
    out = torch.full((B, N, 3), 1e3, dtype=torch.float32, device=device)
    L = torch.zeros(B, dtype=torch.int64, device=device)
    for i, c in enumerate(clouds):
        out[i, :len(c)] = torch.as_tensor(c, dtype=torch.float32, device=device)
        L[i] = len(c)
    return out, L


def centers(c):
    return (np.asarray(c, dtype=np.float32) + 0.5) / GRID - 0.5


def _masked_mean(x, L):
    N = x.shape[1]
    m = (torch.arange(N, device=x.device)[None] < L[:, None]).float()
    return (x * m).sum(1) / L.float()


def chamfer_loss(pred_all, tgt_ds, pred_len, tgt_len, sqrt=False):
    from pytorch3d.loss import chamfer_distance
    loss_tuple, _ = chamfer_distance(pred_all, tgt_ds, x_lengths=pred_len, y_lengths=tgt_len,
                                     batch_reduction=None, point_reduction=None)
    fwd_per_point = loss_tuple[0]
    bwd_per_point = loss_tuple[1]
    if sqrt:
        fwd_per_point = fwd_per_point.clamp_min(0).sqrt()
        bwd_per_point = bwd_per_point.clamp_min(0).sqrt()
    loss = _masked_mean(fwd_per_point, pred_len)
    loss = loss + _masked_mean(bwd_per_point, tgt_len)
    return loss


def cd_batch(A, LA, B, LB):
    return 0.5 * chamfer_loss(A, B, LA, LB, sqrt=True)


def pose24_batch(gen_int, ref_pts, ref_len, device, chunk=8):
    from .contract import rotate, _all_names
    names = _all_names()
    B = len(gen_int)
    best_cd = torch.full((B,), float("inf"), device=device)
    best_k = torch.zeros(B, dtype=torch.int64, device=device)
    cd_raw = torch.zeros(B, device=device)
    rotated = [[rotate(g, n, GRID) for n in names] for g in gen_int]
    for s in range(0, B, chunk):
        idx = list(range(s, min(B, s + chunk)))
        clouds = [centers(rotated[i][k]) for i in idx for k in range(24)]
        P, LP = pad_clouds(clouds, device)
        Q = ref_pts[idx].repeat_interleave(24, dim=0); LQ = ref_len[idx].repeat_interleave(24)
        cd = cd_batch(P, LP, Q, LQ).view(len(idx), 24)
        mn, am = cd.min(dim=1)
        best_cd[idx] = mn; best_k[idx] = am
        cd_raw[idx] = cd[:, names.index("+X+Y+Z")]
    best_names = [names[int(k)] for k in best_k]
    best_int = [rotated[i][int(best_k[i])] for i in range(B)]
    return best_names, best_cd, cd_raw, best_int


def _rot_small(axis, deg, device):
    th = np.deg2rad(deg); c, s = np.cos(th), np.sin(th)
    M = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]], "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
         "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]
    return torch.tensor(M, dtype=torch.float32, device=device)


def _umeyama_batch(A, WA, Bm, WB):
    W = WA * WB
    w = W.sum(1, keepdim=True).clamp_min(1)
    ma = (A * W[..., None]).sum(1) / w; mb = (Bm * W[..., None]).sum(1) / w
    Ac = (A - ma[:, None]) * W[..., None]; Bc = Bm - mb[:, None]
    H = Ac.transpose(1, 2) @ Bc
    U, S, Vt = torch.linalg.svd(H)
    d = torch.sign(torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    dg = torch.stack([torch.ones_like(d), torch.ones_like(d), d], dim=1)
    R = Vt.transpose(1, 2) @ torch.diag_embed(dg) @ U.transpose(1, 2)
    var_a = ((A - ma[:, None]) ** 2 * W[..., None]).sum((1, 2)).clamp_min(1e-12)
    s = (S * dg).sum(1) / var_a
    t = mb - s[:, None] * (R @ ma[..., None])[..., 0]
    return s, R, t


def icp_batch(P, LP, Q, LQ, iters=20):
    B, N, _ = P.shape; M = Q.shape[1]
    mP = (torch.arange(N, device=P.device)[None] < LP[:, None]).float()
    mQ = (torch.arange(M, device=P.device)[None] < LQ[:, None]).float()
    X = P.clone()
    best_cd = best_X = None
    for it in range(iters + 1):
        nxq = knn_points(X, Q, lengths1=LP, lengths2=LQ, K=1)
        nqx = knn_points(Q, X, lengths1=LQ, lengths2=LP, K=1)
        d_xq = torch.where(mP > 0, nxq.dists[..., 0], torch.zeros_like(nxq.dists[..., 0]))
        d_qx = torch.where(mQ > 0, nqx.dists[..., 0], torch.zeros_like(nqx.dists[..., 0]))
        cd = 0.5 * (_masked_mean(d_xq.clamp_min(0).sqrt(), LP) + _masked_mean(d_qx.clamp_min(0).sqrt(), LQ))
        if best_cd is None:
            best_cd, best_X = cd, X.clone()
        else:
            better = cd < best_cd
            best_cd = torch.where(better, cd, best_cd)
            best_X[better] = X[better]
        if it == iters:
            break
        i_xq, i_qx = nxq.idx[..., 0], nqx.idx[..., 0]
        A = torch.cat([X, torch.gather(X, 1, i_qx[..., None].expand(-1, -1, 3))], 1)
        Bm = torch.cat([torch.gather(Q, 1, i_xq[..., None].expand(-1, -1, 3)), Q], 1)
        W = torch.cat([mP, mQ], 1)
        s, R, t = _umeyama_batch(A, W, Bm, W)
        X = s[:, None, None] * (X @ R.transpose(1, 2)) + t[:, None]
    return best_X, best_cd


def icp_multi_batch(P, LP, Q, LQ, deg=4.0, iters=20):
    B = P.shape[0]
    ctr = (P * (torch.arange(P.shape[1], device=P.device)[None] < LP[:, None]).float()[..., None]).sum(1) / LP[:, None].float()
    starts = [P]
    for ax in "xyz":
        for d in (-deg, deg):
            starts.append((P - ctr[:, None]) @ _rot_small(ax, d, P.device).T + ctr[:, None])
    S = len(starts)
    Pall = torch.cat(starts, 0); LPall = LP.repeat(S)
    Qall = Q.repeat(S, 1, 1); LQall = LQ.repeat(S)
    Xb, cdb = icp_batch(Pall, LPall, Qall, LQall, iters)
    cdb = cdb.view(S, B); Xb = Xb.view(S, B, -1, 3)
    mn, am = cdb.min(dim=0)
    return Xb[am, torch.arange(B, device=P.device)], mn


def _rand_sub(X, L, n, seed=0):
    g = np.random.default_rng(seed)
    clouds = []
    for i in range(X.shape[0]):
        c = X[i, :int(L[i])]
        clouds.append(c if len(c) <= n else c[torch.as_tensor(np.sort(g.choice(len(c), n, replace=False)), device=X.device)])
    return pad_clouds([c.cpu().numpy() for c in clouds], X.device)


def align_similarity(P, LP, Q, LQ, align_pts=32768):
    Ps, LPs = _rand_sub(P, LP, align_pts, 0); Qs, LQs = _rand_sub(Q, LQ, align_pts, 1)
    with torch.no_grad():
        Xs, _ = icp_multi_batch(Ps, LPs, Qs, LQs)
        m = (torch.arange(Ps.shape[1], device=P.device)[None] < LPs[:, None]).float()
        s, R, t = _umeyama_batch(Ps, m, Xs, m)
        worse = cd_batch(apply_similarity(P, s, R, t), LP, Q, LQ) >= cd_batch(P, LP, Q, LQ)
        s = torch.where(worse, torch.ones_like(s), s)
        R = torch.where(worse[:, None, None], torch.eye(3, device=P.device)[None], R)
        t = torch.where(worse[:, None], torch.zeros_like(t), t)
    return s, R, t


def apply_similarity(X, s, R, t):
    return s[:, None, None] * (X @ R.transpose(1, 2)) + t[:, None]


_PAD = 16


def iou_levels_batch(a_int, b_int, device, pad=0):
    B = len(a_int); G = GRID + 2 * pad
    def dense(lst):
        d = torch.zeros(B, 1, G, G, G, dtype=torch.float32, device=device)
        for i, c in enumerate(lst):
            t = torch.as_tensor(c, device=device) + pad
            t = t[((t >= 0) & (t < G)).all(1)]
            d[i, 0, t[:, 0], t[:, 1], t[:, 2]] = 1.0
        return d
    A, Bd = dense(a_int), dense(b_int)
    out = {}
    for L in LEVELS:
        k = GRID // L
        Ak = A if k == 1 else torch.nn.functional.max_pool3d(A, k)
        Bk = Bd if k == 1 else torch.nn.functional.max_pool3d(Bd, k)
        inter = (Ak * Bk).flatten(1).sum(1); na = Ak.flatten(1).sum(1); nb = Bk.flatten(1).sum(1)
        out[f"iou{L}"] = inter / (na + nb - inter)
        out[f"prec{L}"] = inter / na
        out[f"rec{L}"] = inter / nb
    return out


def to_index(X):
    return torch.round((X + 0.5) * GRID - 0.5).long()


def align_to_ref(gen_int, ref_int, device, align_pts=32768):
    B = len(gen_int)
    Qf, LQf = pad_clouds([centers(r) for r in ref_int], device)
    with torch.no_grad():
        names, cd_p24, cd_raw, best_int = pose24_batch(gen_int, Qf, LQf, device)
        Pfull, LPfull = pad_clouds([centers(b) for b in best_int], device)
    s, R, t = align_similarity(Pfull, LPfull, Qf, LQf, align_pts)
    with torch.no_grad():
        Xfull = apply_similarity(Pfull, s, R, t)
        idx = [to_index(Xfull[i, :int(LPfull[i])]).cpu().numpy() for i in range(B)]
    return {"pose24": names, "cd_raw": cd_raw, "cd_p24": cd_p24, "best_int": best_int, "s": s, "R": R, "t": t,
            "X": Xfull, "LX": LPfull, "Q": Qf, "LQ": LQf, "idx_refined": idx}


def paired_batch(gen_int, ref_int, device, align_pts=32768):
    B = len(gen_int)
    al = align_to_ref(gen_int, ref_int, device, align_pts)
    with torch.no_grad():
        cd_ref = cd_batch(al["X"], al["LX"], al["Q"], al["LQ"])
        ious = iou_levels_batch(al["best_int"], ref_int, device)
        ious_ref = iou_levels_batch(al["idx_refined"], ref_int, device, pad=_PAD)
    rows = []
    for i in range(B):
        row = {"n_gen": len(gen_int[i]), "n_ref": len(ref_int[i]),
               "cd_raw": float(al["cd_raw"][i]), "cd_p24": float(al["cd_p24"][i]), "pose24": al["pose24"][i],
               "cd_refined": float(cd_ref[i]), "scale_refined": float(al["s"][i])}
        row.update({k: float(v[i]) for k, v in ious.items()})
        row.update({k + "_refined": float(v[i]) for k, v in ious_ref.items()})
        rows.append(row)
    return rows
